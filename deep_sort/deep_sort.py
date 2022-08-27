import numpy as np
import torch
import sys
import cv2
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
from .reid_multibackend import ReIDDetectMultiBackend

__all__ = ['DeepSORT']


class DeepSORT(object):
    def __init__(self, 
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70, n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9
                ):
        
        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)
        self.tracker.center_update()
        
        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def cal_up_down(self, total_up, total_down, frameHeight, im0):
        for track in self.tracker.tracks:
            if len(track.y_centers) > 1 :
                if not track.counted and int(track.y_centers[-1]) in range(frameHeight//2 - 25, frameHeight//2 + 25):
                    direction = track.y_centers[-1] - (sum(track.y_centers[:-1])/len(track.y_centers[:-1]))
                    cv2.circle(im0, (int(track.x_centers[-1]),int(track.y_centers[-1])), 4, (0, 255, 0), -1)
                    if direction < 0 :
                        total_up += 1
                        track.counted = True
                    elif direction > 0 :
                        total_down += 1
                        track.counted = True
                    
        return total_up, total_down, im0
    
    def cal_left_right(self, total_left, total_right, frameWidth, im0):
        for track in self.tracker.tracks:
            if len(track.x_centers) > 1 :
                if not track.counted and int(track.x_centers[-1]) in range(frameWidth//2 - 25, frameWidth//2 + 25):
                    direction = track.x_centers[-1] - (sum(track.x_centers[:-1])/len(track.x_centers[:-1]))
                    cv2.circle(im0, (int(track.x_centers[-1]),int(track.y_centers[-1])), 4, (0, 255, 0), -1)
                    if direction < 0 :
                        total_left += 1
                        track.counted = True
                    elif direction > 0 :
                        total_right += 1
                        track.counted = True
                    
        return total_left, total_right, im0
        
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
