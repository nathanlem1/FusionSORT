"""
For more details about this Gaussian-smoothed interpolation (GSI), please refer to https://arxiv.org/abs/2202.13514
"""
import argparse
import glob
import numpy as np
import os
import sys
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

sys.path.append('.')


def make_parser():
    parser = argparse.ArgumentParser("Interpolation!")
    parser.add_argument("--txt_path", default="", help="path to tracking result path in MOTChallenge format")
    parser.add_argument("--save_path", default=None, help="save result path, none for override")

    return parser


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def write_results_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for i in range(results.shape[0]):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[2:6]
            score = frame_data[6]
            line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
            f.write(line)


# Linear interpolation
def linear_interpolation(input_, interval):
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]  # Sort by ID and frame
    output_ = input_.copy()
    '''Linear interpolation'''
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    for row in input_:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre:  # Same ID
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):  # Frame-by-frame interpolation
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:  # Different ID
            id_pre = id_curr
        row_pre = row
        f_pre = f_curr
    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]
    return output_


# Gaussian smoothing
def gaussian_smooth(input_, tau):
    output_ = list()
    ids = set(input_[:, 1])
    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        t = tracks[:, 0].reshape(-1, 1)
        x = tracks[:, 2].reshape(-1, 1)
        y = tracks[:, 3].reshape(-1, 1)
        w = tracks[:, 4].reshape(-1, 1)
        h = tracks[:, 5].reshape(-1, 1)
        gpr.fit(t, x)
        # xx = gpr.predict(t)[:, 0]
        xx = gpr.predict(t)
        gpr.fit(t, y)
        # yy = gpr.predict(t)[:, 0]
        yy = gpr.predict(t)
        gpr.fit(t, w)
        # ww = gpr.predict(t)[:, 0]
        ww = gpr.predict(t)
        gpr.fit(t, h)
        # hh = gpr.predict(t)[:, 0]
        hh = gpr.predict(t)
        output_.extend([
            [t[i, 0], id_, xx[i], yy[i], ww[i], hh[i], 1, -1, -1 , -1] for i in range(len(t))
        ])
    return output_


def gaussian_smoothed_interpolation(txt_path, save_path, interval=20, tau=10):
    seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.txt')))
    for seq_txt in seq_txts:
        seq_name = seq_txt.split('/')[-1]
        print(seq_name)
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')
        li = linear_interpolation(seq_data, interval)
        gsi = gaussian_smooth(li, tau)
        gsi = np.array(gsi)
        seq_results = gsi[gsi[:, 0].argsort()]
        save_seq_txt = os.path.join(save_path, seq_name)
        write_results_score(save_seq_txt, seq_results)


if __name__ == '__main__':
    args = make_parser().parse_args()

    if args.save_path is None:
        args.save_path = args.txt_path

    mkdir_if_missing(args.save_path)
    gaussian_smoothed_interpolation(args.txt_path, args.save_path, interval=20, tau=10)