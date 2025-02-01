import numpy as np

def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """
    Suppress overlapping detections.
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    
    area = w * h
    if scores is None:
        scores = area

    idxs = np.argsort(scores)
    
    return idxs.tolist()