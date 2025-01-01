import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tracker import matching
from tracker.cmc import CMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter_score import KalmanFilterScore

from fast_reid.fast_reid_interfece import FastReIDInterface

from loguru import logger


class STrack(BaseTrack):
    shared_kalman = KalmanFilterScore()

    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # Wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)  # np.float
        self._score = score
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.reid_status = 0  # Set to 0 if no ReId happens or to 1 if ReId happens.

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha_fixed_emb = 0.95  # 0.90, 0.95
        # self.alpha = 0.9
        self.track_high_thresh = 0.6  # Tracking confidence threshold for the first association

    def update_features(self, feat, alpha=0.9):
        """Dynamic appearance i.e. changing alpha per-frame based on detection confidence score according to
        Deep-ocsort can be used here. Fixed alpha=0.9 leads to standard Exponential Moving Average (EMA)"""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
                    multi_mean[i][8] = 0
                    multi_mean[i][9] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_cmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R10x10 = np.kron(np.eye(5, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R10x10.dot(mean)
                mean[:2] += t
                cov = R10x10.dot(cov).dot(R10x10.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, args, removed_stracks, det_feat):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_score_to_xywhs(self._tlwh, self._score))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        # Do ReId here
        removed_stracks = list(dict.fromkeys(removed_stracks))  # Remove duplicate removed (archived) tracks
        if args.use_reid:
            matches = []
            tracks_del = []
            for track in removed_stracks:
                if (track.end_frame - track.start_frame >= 1) and (frame_id - track.end_frame <= args.window_reid):
                    bool_m, dist = matching.match_features_reid(track.smooth_feat, det_feat, args.reid_thresh)
                    if bool_m:
                        matches.append((track, dist))
                        print('Matched for ReId at frame: ', frame_id)
                else:
                    tracks_del.append(track)

            # Delete out of window tracks and tracks with no update at least once (try at least for 4 times i.e. >=5)
            for t in tracks_del:
                removed_stracks.remove(t)

            # Find the best match
            if len(matches) > 0:
                if len(matches) == 1:
                    first_match = matches[0]
                else:
                    sorted_by_dist = sorted(matches, key=lambda m: m[1])  # sort by the distance
                    first_match = sorted_by_dist[0]
                track_matched = first_match[0]
                track_id_match = track_matched.track_id

                self.features = copy.deepcopy(track_matched.features)
                self.features.append(det_feat)
                self.smooth_feat = copy.deepcopy(track_matched.smooth_feat)
                self.update_features(det_feat)

                self.track_id = track_id_match  # Update track_id if reid is successful
                self.reid_status = 1  # ReId happened.

                # Remove the matched track from the archive tracks (removed_stracks)
                removed_stracks.remove(track_matched)

        return removed_stracks

    def re_activate(self, new_track, frame_id, new_id=False):

        self.score = new_track.score
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_score_to_xywhs(new_track.tlwh, new_track.score))  # NSA doesn't help!

        # EMA with dynamic or changing alpha
        trust = (new_track.score - self.track_high_thresh) / (1 - self.track_high_thresh)
        af = self.alpha_fixed_emb
        alpha_ema = af + (1 - af) * (1 - trust)
        if new_track.curr_feat is not None:
            # self.update_features(new_track.curr_feat, alpha_ema)  # Changing alpha doesn't help!
            self.update_features(new_track.curr_feat)  # with default alpha=0.9

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        new_score = new_track.score

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_score_to_xywhs(new_tlwh, new_score))  # NSA doesn't help!

        # EMA with dynamic or changing alpha
        trust = (new_track.score - self.track_high_thresh) / (1 - self.track_high_thresh)
        af = self.alpha_fixed_emb
        alpha_ema = af + (1 - af) * (1 - trust)
        if new_track.curr_feat is not None:
            # self.update_features(new_track.curr_feat, alpha_ema)  #  Changing alpha doesn't help!
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def tlwh_score_to_xywhs(tlwh, score):
        """Convert bounding box and score to format `(center x, center y, width,
        height, score)`.
        """
        meas = np.zeros(5,)
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        meas[:4] = ret
        meas[4] = score
        return meas

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    def to_xywhs(self):
        return self.tlwh_score_to_xywhs(self.tlwh, self.score)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class FusionSORT(object):
    def __init__(self, args, frame_rate=30, num_frames=500):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack] # This is archived tracks.
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh  # Score threshold for first association.
        self.track_low_thresh = args.track_low_thresh    # Score threshold for removing or rejecting bad detections.
        self.new_track_thresh = args.new_track_thresh    # Detection score threshold for creating new tracks.

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilterScore()

        # ReID module
        self.iou_thresh = args.iou_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_appearance:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)
            logger.info("ReID Model Summary: {}".format(self.encoder.model_info))

        self.cmc = CMC(benchmark=args.benchmark, method=args.cmc_method, verbose=[args.name, args.ablation])

    def step(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []  # tracks with re-assigned detection after being lost in the previous frame(s).
        lost_stracks = []   # inactive tracks, i.e., tracks without assigned detection.
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1,y1,x2,y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]  # dets_first
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_appearance:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_appearance:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]  # detections_first
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]   # detections_first
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        if self.args.cmc_method != 'none':
            warp = self.cmc.apply(img, dets)
            STrack.multi_cmc(strack_pool, warp)
            STrack.multi_cmc(unconfirmed, warp)

        # Associate with high score detection boxes
        if self.args.fusion_method != 'kf_gating':
            ious_dists = matching.iou_distance(strack_pool, detections)
            ious_dists_mask = (ious_dists > self.iou_thresh)

        if self.args.with_hiou:
            hious_dists = matching.hiou_distance(strack_pool, detections)
        if self.args.with_confidence:
            confidence_dists = matching.confidence_distance(strack_pool, detections)

        if self.frame_id == 10:
            pass

        if not self.args.mot20:
            if self.args.fusion_method != 'kf_gating':
                ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.fusion_method == 'minimum':
            if self.args.with_appearance:
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)  # Taking MINIMUM of each element as IoU-Appearance fusion.
                if self.args.with_hiou:
                    hious_dists[ious_dists_mask] = 1.0
                    dists = np.minimum(dists, hious_dists)
                if self.args.with_confidence:
                    confidence_dists[ious_dists_mask] = 1.0
                    dists = np.minimum(dists, confidence_dists)
            else:
                dists = ious_dists
        elif self.args.fusion_method == 'weighted_sum':
            if self.args.with_appearance:
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                # emb_dists[ious_dists_mask] = 1.0
                dists = self.args.lambda1 * ious_dists + self.args.lambda2 * emb_dists
                if self.args.with_hiou:
                    # hious_dists[ious_dists_mask] = 1.0
                    dists = dists + self.args.lambda3 * hious_dists
                if self.args.with_confidence:
                    # confidence_dists[ious_dists_mask] = 1.0
                    dists = dists + self.args.lambda4 * confidence_dists
            else:
                dists = ious_dists
        elif self.args.fusion_method == 'kf_gating':
            dists = matching.mahalanobis_distance(self.kalman_filter, strack_pool, detections)
            if self.args.with_appearance:
                raw_emb_dists = matching.embedding_distance(strack_pool, detections)
                if self.args.with_hiou:
                    raw_emb_dists = raw_emb_dists + self.args.lambda_h_kf * hious_dists
                if self.args.with_confidence:
                    raw_emb_dists = raw_emb_dists + self.args.lambda_c_kf * confidence_dists
                dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
        elif self.args.fusion_method == 'hadamard':
            if self.args.with_appearance:
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = ious_dists * emb_dists
                if self.args.with_hiou:
                    hious_dists[ious_dists_mask] = 1.0
                    dists = dists * hious_dists
                if self.args.with_confidence:
                    confidence_dists[ious_dists_mask] = 1.0
                    dists = dists * confidence_dists
            else:
                dists = ious_dists
        else:
            raise ValueError('Set to correct fusion method: minimum, weighted_sum, kf_gating, or hadamard ')

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)  # u_detection_first

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)   # Apply Kalman filter update
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)  # Re-activate and apply Kalman filter update
                refind_stracks.append(track)  # lost tracks matched with detection (within self.max_time_lost)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # Associate the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]  # lost_tracks (inactive tracks) are not considered in the 2nd association.
        if self.args.second_matching_distance == 'iou':
            # IoU is used for all fusion methods for 2nd association
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
        elif self.args.second_matching_distance == 'mahalanobis':
            dists = matching.mahalanobis_distance(self.kalman_filter, r_tracked_stracks, detections_second)
        else:
            raise ValueError('Select the correct matching distance method: iou or mahalanobis')

        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)  # emb_dists doesn't help!
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)   # Apply Kalman filter update
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        if self.args.fusion_method != 'kf_gating':
            ious_dists = matching.iou_distance(unconfirmed, detections)
            ious_dists_mask = (ious_dists > self.iou_thresh)

        if self.args.with_hiou:
            hious_dists = matching.hiou_distance(unconfirmed, detections)
        if self.args.with_confidence:
            confidence_dists = matching.confidence_distance(unconfirmed, detections)

        if not self.args.mot20:
            if self.args.fusion_method != 'kf_gating':
                ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.fusion_method == 'minimum':
            if self.args.with_appearance:
                emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)  # Taking MINIMUM of each element as IoU-Appearance fusion.
                if self.args.with_hiou:
                    hious_dists[ious_dists_mask] = 1.0
                    dists = np.minimum(dists, hious_dists)
                if self.args.with_confidence:
                    confidence_dists[ious_dists_mask] = 1.0
                    dists = np.minimum(dists, confidence_dists)
            else:
                dists = ious_dists
        elif self.args.fusion_method == 'weighted_sum':
            if self.args.with_appearance:
                emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                # emb_dists[ious_dists_mask] = 1.0   # Using mask due to IoU decreases result for this fusion method
                dists = self.args.lambda1 * ious_dists + self.args.lambda2 * emb_dists
                if self.args.with_hiou:
                    # hious_dists[ious_dists_mask] = 1.0
                    dists = dists + self.args.lambda3 * hious_dists
                if self.args.with_confidence:
                    # confidence_dists[ious_dists_mask] = 1.0
                    dists = dists + self.args.lambda4 * confidence_dists
            else:
                dists = ious_dists
        elif self.args.fusion_method == 'kf_gating':
            dists = matching.mahalanobis_distance(self.kalman_filter, unconfirmed, detections)
            if self.args.with_appearance:
                raw_emb_dists = matching.embedding_distance(unconfirmed, detections)
                if self.args.with_hiou:
                    raw_emb_dists = raw_emb_dists + self.args.lambda_h_kf * hious_dists
                if self.args.with_confidence:
                    raw_emb_dists = raw_emb_dists + self.args.lambda_c_kf * confidence_dists
                dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, unconfirmed, detections)
        elif self.args.fusion_method == 'hadamard':
            if self.args.with_appearance:
                emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = ious_dists * emb_dists
                if self.args.with_hiou:
                    hious_dists[ious_dists_mask] = 1.0
                    dists = dists * hious_dists
                if self.args.with_confidence:
                    confidence_dists[ious_dists_mask] = 1.0
                    dists = dists * confidence_dists
            else:
                dists = ious_dists
        else:
            raise ValueError('Set to correct fusion method: minimum, weighted_sum, kf_gating, or hadamard ')

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)  # Apply Kalman filter update
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            self.removed_stracks = track.activate(self.kalman_filter, self.frame_id, self.args, self.removed_stracks,
                                                  track.curr_feat)  # Initiate new tracks. Include ReId here.

            activated_starcks.append(track)

        """ Step 5: Manage lost tracks """
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]  # Note that we do not output the boxes and
        # identities of lost tracks.

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
