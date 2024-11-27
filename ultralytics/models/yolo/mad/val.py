# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, MADMetrics, box_iou, kpt_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class MADValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a MAD model.

    Example:
        ```python
        from ultralytics.models.yolo.mad import MADValidator

        args = dict(model="yolov8n-mad.pt", data="coco8-MAD.yaml")
        validator = MADValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize a 'MADValidator' object with custom parameters and assigned attributes."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.sigma = None
        self.nattr = None
        self.args.task = "mad"
        self.metrics = MADMetrics(save_dir=self.save_dir, on_plot=self.on_plot)


    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'attrs' data into a float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["attributes"] = batch["attributes"].to(self.device).float()
        return batch

    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Attr(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """Apply non-maximum suppression and return detections with high confidence scores."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            rotated=True,
        )
        
        for pred in preds:
            pred[:, 6:] = pred[:, 6:].sigmoid() 
            
        return preds

    def init_metrics(self, model):
        """Initiate MAD metrics for YOLO model."""
        super().init_metrics(model)
        self.nattr = self.data["nattr"]
        self.stats = dict(tp_a=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def _prepare_batch(self, si, batch):
        """Prepares a batch for processing by converting attributes to float and moving to device."""
        pbatch = super()._prepare_batch(si, batch)
        attributes = batch["attributes"][batch["batch_idx"] == si]
        pbatch["attributes"] = attributes
        return pbatch

    def _prepare_pred(self, pred, pbatch):
        """Prepares and scales keypoints in a batch for pose processing."""
        predn = super()._prepare_pred(pred, pbatch)

        # Extract and reshape binary attributes from predictions
        pred_attrs = predn[:, -self.nattr:].view(len(predn), self.nattr, -1)  # Extract last `nattr` dimensions

        return predn, pred_attrs

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_a=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_attrs = self._prepare_pred(pred, pbatch)
            pred_attrs = pred_attrs.squeeze(-1)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_a"] = self._process_batch(predn, bbox, cls, pred_attrs, pbatch["attributes"])
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_attrs,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )


    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_attrs=None, gt_attrs=None):
        """
        Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground truth,
        and finding true positives for multi-label (binary) classification.

        Args:
            detections (torch.Tensor): Tensor with shape (N, 6) representing detection boxes and scores, where each
                detection is of the format (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor with shape (M, 4) representing ground truth bounding boxes, where each
                box is of the format (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor with shape (M,) representing ground truth class indices.
            pred_attrs (torch.Tensor | None): Optional tensor with shape (N, K) representing predicted binary attributes
                for K attributes.
            gt_attrs (torch.Tensor | None): Optional tensor with shape (M, K) representing ground truth binary attributes
                for K attributes.

        Returns:
            torch.Tensor: A tensor with shape (N, 10) representing the correct prediction matrix for 10 IoU levels,
                where N is the number of detections.

        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        
        if pred_attrs is not None and gt_attrs is not None:
            pred_attrs = pred_attrs > 0.5
            correct_attrs = (gt_attrs[:, None, :] == pred_attrs[None, :, :]).all(dim=-1)
            iou = iou * correct_attrs
        
        return self.match_predictions(detections[:, 5], gt_cls, iou)


    def plot_val_samples(self, batch, ni):
        """Plots and saves validation set samples with predicted bounding boxes and keypoints."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            attrs=batch['attributes'],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predictions for YOLO model."""
        pred_attrs = torch.cat([p[:, 6:].view(-1, self.nattr) for p in preds], 0)
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            attrs=pred_attrs,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, pred_attrs, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            attributes=pred_attrs,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Converts YOLO predictions to COCO JSON format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "attributes": p[6:],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics for multiple binary attributes."""
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions
            pred_txt = self.save_dir / "predictions_txt"  # predictions
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))  # Load prediction data in JSON format

            # Save split results for multiple binary attributes
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"]].replace(" ", "-")
                attributes = d["attributes"]  # Assuming 'attributes' is a list of binary values (e.g., [1, 0, 1, 0])

                # Save the attributes and score
                with open(f'{pred_txt / f"Task1_{classname}"}.txt', "a") as f:
                    attribute_str = " ".join(map(str, attributes))  # Convert the list of attributes to a space-separated string
                    f.writelines(f"{image_id} {score} {attribute_str}\n")  # Save image_id, score, and attributes as a string

            # Save merged results for multiple binary attributes (if necessary)
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # merged predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)

            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__")[0]
                attributes = d["attributes"]  # List of binary attributes
                score = d["score"]
                cls = d["category_id"]

                # Append attributes, score, and class to merged results
                merged_results[image_id].append([attributes, score, cls])

            # Process merged results and save them
            for image_id, results in merged_results.items():
                for result in results:
                    attributes, score, cls = result
                    classname = self.names[cls].replace(" ", "-")

                    # Save the merged prediction results for multiple binary attributes
                    with open(f'{pred_merged_txt / f"Task1_{classname}"}.txt', "a") as f:
                        attribute_str = " ".join(map(str, attributes))  # Convert the list of attributes to a space-separated string
                        f.writelines(f"{image_id} {score} {attribute_str}\n")  # Save image_id, score, and attributes

        return stats

