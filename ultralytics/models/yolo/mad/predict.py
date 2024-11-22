# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class MADPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a MAD model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.mad import MADPredictor

        args = dict(model="yolov8n-mad.pt", source=ASSETS)
        predictor = MADPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes MADPredictor, sets task to 'MAD'"""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "MAD"
        
    def postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images, with multiple binary attributes."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            attrs = pred[:, 6:].int()  

            # Construct results with attributes, score, and boxes
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], attributes=attrs)
            )
        
        return results

