# vim: expandtab:ts=4:sw=4
# Implementation of vectorized version of the Kalman filter.
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilterScore(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 10-dimensional state space

        x, y, w, h, c, vx, vy, vw, vh, vc

    contains the bounding box center position (x, y), width w, height h, tracklet confidence (score) c
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location and detection score
    (x, y, w, h, c) is taken as direct observation of the state space (linear observation model).

    """

    def __init__(self):
        ndim, dt = 5, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)   # F_k
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)   # H_k

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, w, h, c) with center position (x, y),
            width w, height h and tracklet confidence.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (10 dimensional) and covariance matrix (10x10
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            20 * self._std_weight_position * measurement[4],  # for tracklet score or confidence c
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            100 * self._std_weight_velocity * measurement[4]  # for tracklet score or confidence velocity cv
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 10 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 10x10 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[4]]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[4]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))  # Q_k

        mean = np.dot(mean, self._motion_mat.T)   # F_k x_k_1
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov  # F_k P_k_1 F_k^T + Q_k

        return mean, covariance

    def project(self, mean, covariance, det_score=None):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (10 dimensional array).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).
        det_score : float
            Detection confidence score. This is used for Noise Scale Adaptive (NSA) Kalman Filter i.e.
            R_k = (1 -score) R_k, Refer StrongSORT for more details about NSA KF.

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            2 * self._std_weight_position * mean[4]  # for tracklet score or confidence c
        ]
        if det_score is not None:
            std = [(1. - det_score) * x for x in std]   # If score is not None, NSA KF is used.
        innovation_cov = np.diag(np.square(std))  # R_k

        mean = np.dot(self._update_mat, mean)     # H_k x_k|k_1
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))  # H_k P_k|k-1 H_k^T
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx10 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx10x10 dimensional covariance matrices of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 4]]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 4]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement, det_score=None):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (10 dimensional).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).
        measurement : ndarray
            The 5 dimensional measurement vector (x, y, w, h, c), where (x, y)
            is the center position, w the width, h the height and c detection score of the
            bounding box.
        det_score : float
            Detection confidence score. This is used for noise scale adaptive (NSA) Kalman Filter i.e.
            R_k = (1 -score) R_k.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance, det_score)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean    # Z_k - H_k x_k|k_1

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (10 dimensional).
        covariance : ndarray
            Covariance of the state distribution (10x10 dimensional).
        measurements : ndarray
            An Nx5 dimensional matrix of N measurements, each in
            format (x, y, w, h, c) where (x, y) is the bounding box center
            position, w is the width, h is the height and c is the detection score.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
