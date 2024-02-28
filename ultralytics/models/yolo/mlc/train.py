# Ultralytics YOLO 🚀, AGPL-3.0 license
import os

import torch
import torchvision
import torchvision.transforms as transforms

from ultralytics.data import (
    MultiLabelClassificationDataset,
    build_dataloader,
)
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import MLCModel, attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import (
    is_parallel,
    strip_optimizer,
    torch_distributed_zero_first,
)
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class MLCTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.models.yolo.classify import ClassificationTrainer

        args = dict(model='yolov8n-cls.pt', data='imagenet10', epochs=3)
        trainer = ClassificationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a ClassificationTrainer object with optional configuration overrides and callbacks."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "mlc"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)
        self.base_path = self.data["path"]
        self.attributes = list(self.data["names"].values())

    def set_model_attributes(self):
        """Set the YOLO model's class names from the loaded dataset."""
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""
        model = MLCModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

    def setup_model(self):
        """Load, create or download model for any task."""
        if isinstance(
            self.model, torch.nn.Module
        ):  # if model is loaded beforehand. No setup needed
            return

        model, ckpt = str(self.model), None
        # Load a YOLO model locally, from torchvision, or from Ultralytics assets
        if model.endswith(".pt"):
            self.model, ckpt = attempt_load_one_weight(model, device="cpu")
            for p in self.model.parameters():
                p.requires_grad = True  # for training
        elif model.split(".")[-1] in ("yaml", "yml"):
            self.model = self.get_model(cfg=model)
        elif model in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
        else:
            FileNotFoundError(
                f"ERROR: model={model} not found locally or online. Please check model name."
            )
        MLCModel.reshape_outputs(self.model, self.data["nc"])

        return ckpt

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return MultiLabelClassificationDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            cache=self.args.cache or None,
            stride=int(gs),
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns PyTorch DataLoader with transforms to preprocess images for inference."""
        with torch_distributed_zero_first(
            rank
        ):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(
                img_path=dataset_path, mode=mode, batch=batch_size
            )

        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)
        # Attach inference transforms
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.transforms
            else:
                self.model.transforms = loader.dataset.transforms
        return loader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """Returns an instance of MLCValidator for validation."""
        self.loss_names = ["loss"]
        return yolo.mlc.MLCValidator(
            self.test_loader, self.save_dir, _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(
            file=self.csv, classify=True, on_plot=self.on_plot
        )  # save results.png

    def final_eval(self):
        """Evaluate trained model and save validation results."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=[
                torch.nonzero(sample > 0.5, as_tuple=True)[0].cpu().numpy()
                for sample in batch["cls"]
            ],  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
