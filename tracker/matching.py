import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracker import kalman_filter_score


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def intersection_batch(bboxes_1, bboxes_2):
    """
    Computes INTERSECTION between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes_2 = np.expand_dims(bboxes_2, 0)
    bboxes_1 = np.expand_dims(bboxes_1, 1)

    xx1 = np.maximum(bboxes_1[..., 0], bboxes_2[..., 0])
    yy1 = np.maximum(bboxes_1[..., 1], bboxes_2[..., 1])
    xx2 = np.minimum(bboxes_1[..., 2], bboxes_2[..., 2])
    yy2 = np.minimum(bboxes_1[..., 3], bboxes_2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersections = w * h
    return intersections


def box_area(bbox):
    """
    Computes AREA of a bbox in the form [x1,y1,x2,y2]
    """
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return area


def ious(atlbrs, btlbrs):
    """
    Compute Intersection-Over-Union (IoU) of two bounding boxes.
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)  # np.float
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def iou_batch(bboxes_1, bboxes_2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes_2 = np.expand_dims(bboxes_2, 0)
    bboxes_1 = np.expand_dims(bboxes_1, 1)

    xx1 = np.maximum(bboxes_1[..., 0], bboxes_2[..., 0])
    yy1 = np.maximum(bboxes_1[..., 1], bboxes_2[..., 1])
    xx2 = np.minimum(bboxes_1[..., 2], bboxes_2[..., 2])
    yy2 = np.minimum(bboxes_1[..., 3], bboxes_2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    _ious = wh / ((bboxes_1[..., 2] - bboxes_1[..., 0]) * (bboxes_1[..., 3] - bboxes_1[..., 1]) +
                  (bboxes_2[..., 2] - bboxes_2[..., 0]) * (bboxes_2[..., 3] - bboxes_2[..., 1]) - wh)
    return _ious


def hmiou(atlbrs, btlbrs):
    """  Height_Modulated_IoU

    :param atlbrs: bbox a (N,4)(x1,y1,x2,y2)
    :param btlbrs: bbox b (N,4)(x1,y1,x2,y2)
    :return:
        height modulated iou (hmiou)
    """
    hm_ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if hm_ious.size == 0:
        return hm_ious
    btlbrs = np.expand_dims(btlbrs, 0)
    atlbrs = np.expand_dims(atlbrs, 1)

    yy11 = np.maximum(atlbrs[..., 1], btlbrs[..., 1])
    yy12 = np.minimum(atlbrs[..., 3], btlbrs[..., 3])

    yy21 = np.minimum(atlbrs[..., 1], btlbrs[..., 1])
    yy22 = np.maximum(atlbrs[..., 3], btlbrs[..., 3])
    h_iou = (yy12 - yy11) / (yy22 - yy21)  # height IoU

    xx1 = np.maximum(atlbrs[..., 0], btlbrs[..., 0])
    yy1 = np.maximum(atlbrs[..., 1], btlbrs[..., 1])
    xx2 = np.minimum(atlbrs[..., 2], btlbrs[..., 2])
    yy2 = np.minimum(atlbrs[..., 3], btlbrs[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((atlbrs[..., 2] - atlbrs[..., 0]) * (atlbrs[..., 3] - atlbrs[..., 1]) +
                (btlbrs[..., 2] - btlbrs[..., 0]) * (btlbrs[..., 3] - btlbrs[..., 1]) - wh)  # conventional IoU

    hm_ious = iou * h_iou  # height-modulated IoUs
    return hm_ious


def hiou(atlbrs, btlbrs):
    """  Height_Modulated_IoU

    :param atlbrs: bbox a (N,4)(x1,y1,x2,y2)
    :param btlbrs: bbox b (N,4)(x1,y1,x2,y2)
    :return:
        height iou (hiou)
    """
    h_ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if h_ious.size == 0:
        return h_ious
    btlbrs = np.expand_dims(btlbrs, 0)
    atlbrs = np.expand_dims(atlbrs, 1)

    yy11 = np.maximum(atlbrs[..., 1], btlbrs[..., 1])
    yy12 = np.minimum(atlbrs[..., 3], btlbrs[..., 3])

    yy21 = np.minimum(atlbrs[..., 1], btlbrs[..., 1])
    yy22 = np.maximum(atlbrs[..., 3], btlbrs[..., 3])
    h_ious = (yy12 - yy11) / (yy22 - yy21)  # height IoUs
    return h_ious


def tlbr_expand(tlbr, scale=1.2):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)  # iou similarity
    cost_matrix = 1 - _ious  # cost

    return cost_matrix


def hmiou_distance(atracks, btracks):
    """
    Compute cost based on height-modulated IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    _hmious = hmiou(atlbrs, btlbrs)
    cost_matrix = 1 - _hmious

    return cost_matrix


def hiou_distance(atracks, btracks):
    """
    Compute cost based on height IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    _hious = hiou(atlbrs, btlbrs)
    cost_matrix = 1 - _hious
    if cost_matrix.size != 0:
        cost_matrix = cost_matrix / np.max(cost_matrix)  # Normalize to [0, 1]
    return cost_matrix


def confidence_distance(atracks, btracks):
    """
    Compute cost based on track score
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    _score_diffs = np.zeros((len(atracks), len(btracks)), dtype=float)
    if _score_diffs.size == 0:
        return _score_diffs

    ascores = [track.mean[4] for track in atracks]   # Corresponds to tracks; estimated tracklet confidence is used
    bscores = [track.score for track in btracks]   # Corresponds to detections, detection score is used.

    bscores = np.expand_dims(bscores, 0)
    ascores = np.expand_dims(ascores, 1)

    _score_diffs = np.abs(ascores - bscores)
    cost_matrix = _score_diffs / np.max(_score_diffs)  # Normalize to [0, 1]

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)

    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def cosine_distance(features1, features2):
    dot_product_f1f2 = np.dot(features1, features2.T)
    norm_feat1 = np.linalg.norm(features1)
    norm_feat2 = np.linalg.norm(features2)
    cosine_dist = dot_product_f1f2 / (norm_feat1 * norm_feat2)
    return cosine_dist


def match_features_reid(feature1, feature2, threshold):
    cosine_dist = cosine_distance(feature1, feature2)
    dist = (1 - cosine_dist) / 2.0   # 0 - very similar, 1 - very different
    if dist <= threshold:  # threshold = 0.3 or 0.2
        return True, dist
    else:
        return False, dist


def mahalanobis_distance(kf, tracks, detections, only_position=True):
    """
    Compute cost based on track score
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    _mahalanobis_dists = np.zeros((len(tracks), len(detections)), dtype=float)
    if _mahalanobis_dists.size == 0:
        return _mahalanobis_dists

    # gating_dim = 2 if only_position else 4  # if measurement is [x, y, w, h] or [x, y, a, h]
    gating_dim = 2 if only_position else 5  # if measurement is [x, y, w, h, s] where s is score or confidence
    gating_threshold = kalman_filter_score.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    # measurements = np.asarray([det.to_xywh() for det in detections])
    measurements = np.asarray([det.to_xywhs() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        _mahalanobis_dists[row, gating_distance > gating_threshold] = np.inf
    cost_matrix = _mahalanobis_dists
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=True, lambda_=0.98):
    """ Fusion method (specially used in JDE / FairMOT) """
    if cost_matrix.size == 0:
        return cost_matrix
    # gating_dim = 2 if only_position else 4  # if measurement is [x, y, w, h] or [x, y, a, h]
    gating_dim = 2 if only_position else 5   # if measurement is [x, y, w, h, s] where s is score or confidence
    gating_threshold = kalman_filter_score.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    # measurements = np.asarray([det.to_xywh() for det in detections])
    measurements = np.asarray([det.to_xywhs() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix  # iou similarity
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores  # iou similarity is weighted by detection scores.
    fuse_cost = 1 - fuse_sim
    return fuse_cost
