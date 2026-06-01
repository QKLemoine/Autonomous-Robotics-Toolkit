"""
Evaluation metrics for 3D Instance Segmentation.

Implements Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).
Utilizes Hungarian matching to optimally assign predictions to ground truth labels.
"""

from typing import Dict, Any, List
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0.0 if union == 0 else float(intersection) / float(union)


def evaluate_panoptic_quality(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5,
    gt_bg_labels: List[int] = [0, 1, 2]
) -> Dict[str, Any]:
    """
    Evaluates instance matching by IoU to calculate Panoptic Quality.
    """
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]
    
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[~np.isin(gt_ids, gt_bg_labels)] 
    
    num_pred = len(pred_ids)
    num_gt = len(gt_ids)
    
    if num_pred == 0 and num_gt == 0:
        return {'pq': 1.0, 'sq': 1.0, 'rq': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    
    if num_pred == 0 or num_gt == 0:
        return {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'tp': 0, 'fp': num_pred, 'fn': num_gt}
    
    LARGE_COST = 1e9  
    cost_matrix = np.full((num_pred, num_gt), LARGE_COST, dtype=np.float64)
    iou_matrix = np.zeros((num_pred, num_gt), dtype=np.float64)
    
    for i, pid in enumerate(pred_ids):
        pred_binary = (pred_mask == pid)
        for j, gid in enumerate(gt_ids):
            gt_binary = (gt_mask == gid)
            iou = compute_iou(pred_binary, gt_binary)
            iou_matrix[i, j] = iou
            
            if iou >= iou_threshold:
                cost_matrix[i, j] = 1.0 - iou
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matched_ious = []
    matched_gt = set()
    
    for r, c in zip(row_ind, col_ind):
        iou = iou_matrix[r, c]
        if iou >= iou_threshold:
            matched_ious.append(iou)
            matched_gt.add(gt_ids[c])
    
    tp = len(matched_ious)
    fp = num_pred - tp
    fn = num_gt - len(matched_gt)
    
    sq = np.mean(matched_ious) if matched_ious else 0.0
    denominator = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denominator if denominator > 0 else 0.0
    pq = sq * rq
    
    return {
        'pq': round(pq, 4), 
        'sq': round(sq, 4), 
        'rq': round(rq, 4),
        'tp': tp, 
        'fp': fp, 
        'fn': fn
    }