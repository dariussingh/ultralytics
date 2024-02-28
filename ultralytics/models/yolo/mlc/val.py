# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torchvision.transforms as transforms

from ultralytics.data import (
    MultiLabelClassificationDataset,
    build_dataloader,
)
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import MultiLabelClassifyMetrics
from ultralytics.utils.plotting import plot_images


class MLCValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.models.yolo.classify import ClassificationValidator

        args = dict(model='yolov8n-cls.pt', data='imagenet10')
        validator = ClassificationValidator(args=args)
        validator()
        ```
    """

    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
    ):
        """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "classify"
        self.metrics = MultiLabelClassifyMetrics()
        if self.data:
            self.base_path = self.data["path"]
            self.attributes = list(self.data["names"].values())

    def get_desc(self):
        """Returns a formatted string summarizing classification metrics."""
        return ("%22s" + "%11s" * 4) % (
            "classes",
            "acc",
            "precision",
            "recall",
            "mAP",
        )

    def init_metrics(self, model):
        """Initialize confusion matrix, class names, and top-1 and top-5 accuracy."""
        self.names = model.names
        self.nc = len(model.names)
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""

        self.pred.append(preds)
        self.targets.append(batch["cls"])

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes metrics of the model such as confusion_matrix and speed."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_stats(self):
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return MultiLabelClassificationDataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=False
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for classification tasks with given parameters."""
        dataset = MultiLabelClassificationDataset(
            img_path=dataset_path, mode="val", batch=batch_size
        )

        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """Prints evaluation metrics for YOLO object detection model."""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(
            pf
            % (
                "all",
                self.metrics.avg_acc,
                self.metrics.avg_precision,
                self.metrics.avg_recall,
                self.metrics.avg_map,
            )
        )

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=[
                torch.nonzero(sample > 0.5, as_tuple=True)[0].cpu().numpy()
                for sample in batch["cls"]
            ],  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=[
                torch.nonzero(sample > 0.5, as_tuple=True)[0].cpu().numpy()
                for sample in preds
            ],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
