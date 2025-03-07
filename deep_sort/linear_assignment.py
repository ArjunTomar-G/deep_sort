import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

# Precomputed Chi-square inverse CDF values for gating
chi2inv95 = {
    1: chi2.ppf(0.95, df=1),  # 1D position gating
    2: chi2.ppf(0.95, df=2),  # 2D position gating
    3: chi2.ppf(0.95, df=3),  # 3D position gating
    4: chi2.ppf(0.95, df=4)   # 4D full-state gating
}

INFTY_COST = 1e+5 
def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def matching_cascade(
    distance_metric, max_distance, cascade_depth, tracks, detections,
    track_indices=None, detection_indices=None):
    """
    Run matching cascade.
    
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]] -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth : int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following elements:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # No tracks left
            continue

        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections

def min_cost_matching(
    distance_metric, max_distance, tracks, detections, track_indices, detection_indices):
    """
    Solve linear assignment problem.
    """
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    indices = linear_assignment(cost_matrix)
    
    matches, unmatched_tracks, unmatched_detections = [], [], []
    
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
            
    return matches, unmatched_tracks, unmatched_detections

def gate_cost_matrix(
    kf, cost_matrix, tracks, detections, track_indices, detection_indices,
    gated_cost=INFTY_COST, only_position=False):
    """
    Gate cost matrix based on Mahalanobis distance.
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost

    return cost_matrix