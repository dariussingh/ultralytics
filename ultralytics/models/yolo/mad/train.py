# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import MADModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results


class MADTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a multi-attribute detection model.

    Example:
        ```python
        from ultralytics.models.yolo.mad import MADTrainer

        args = dict(model="yolov8n-mad.pt", data="coco8-MAD.yaml", epochs=3)
        trainer = MADTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a PoseTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "mad"
        super().__init__(cfg, overrides, _callbacks)


    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get pose estimation model with specified configuration and weights."""
        model = MADModel(cfg, ch=3, nc=self.data["nc"], data_nattr=self.data["nattr"], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets num attributes for MADModel."""
        super().set_model_attributes()
        self.model.nattr = self.data["nattr"]

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "attr_loss", "cls_loss", "dfl_loss"
        return yolo.mad.MADValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    # def plot_metrics(self):
    #     """Plots training/val metrics."""
    #     plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png
