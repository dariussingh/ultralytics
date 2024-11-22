# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import MLDModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results


class MLDTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a multilabel detection model.

    Example:
        ```python
        from ultralytics.models.yolo.mld import MLDTrainer

        args = dict(model="yolov8n-mld.pt", data="coco8-mld.yaml", epochs=3)
        trainer = MLDTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a PoseTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "mld"
        super().__init__(cfg, overrides, _callbacks)


    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get pose estimation model with specified configuration and weights."""
        model = MLDModel(cfg, ch=3, nc=self.data["nc"], data_nlbl=self.data["nlbl"], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets num labels attribute of MLDModel."""
        super().set_model_attributes()
        self.model.nlbl = self.data["nlbl"]

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "lbl_loss", "cls_loss", "dfl_loss"
        return yolo.mld.MLDValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    # def plot_metrics(self):
    #     """Plots training/val metrics."""
    #     plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png
