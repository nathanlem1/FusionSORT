import argparse
import os
import sys
import os.path as osp
import cv2
import numpy as np
import torch

sys.path.append('.')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess

from tracker.fusion_sort import FusionSORT
from tracker.tracking_utils.timer import Timer
from tracker.tracking_utils.visualization import plot_tracking

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser(description='Tracking pipeline for FusionSORT Tracker.')

    parser.add_argument("--path", default='./datasets/MOT17', type=str,
                        help="path to dataset under evaluation, currently only support MOT17, MOT20 and DanceTrack.")
    parser.add_argument('--output_dir', type=str, default='./results/trackers',
                        help='Path to base tracking result folder to be saved to.')
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17',
                        help="benchmark to evaluate: MOT17 | MOT20 | DanceTrack")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='val',
                        help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default='FSORT1',
                        help='The name of the experiment, used for running different experiments and then evaluations, '
                             'e.g. FSORT1, FSORT2, etc.')
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true",
                        help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true",
                        help="save sequences with tracks.")
    parser.add_argument('--display_tracks', default=False, action="store_true", help='Display sequences with tracks.')

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision evaluation.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6,
                        help="tracking confidence threshold for the first association")
    parser.add_argument("--track_low_thresh", default=0.1, type=float,
                        help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument('--with_nsa', default=False, action='store_true',
                        help='For using Noise Scale Adaptive (NSA) Kalman Filter (R_nsa = (1-detection score)R')

    # CMC
    parser.add_argument("--cmc-method", default="file", type=str,
                        help="cmc method: file | sparseOptFlow | orb | sift | ecc | none")

    # Appearance, ReId and weak cues
    parser.add_argument("--with-appearance", dest="with_appearance", default=False, action="store_true",
                        help="For using appearance representation features.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=None, type=str,
                        help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=None, type=str,
                        help="reid config file path")
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    parser.add_argument('--with-hiou', dest='with_hiou', default=False, action='store_true',
                        help='For using weak clue, particularly height-IoU distance.')
    parser.add_argument('--with-confidence', dest='with_confidence', default=False, action='store_true',
                        help='For using weak clue, particularly confidence distance.')

    parser.add_argument("--use-reid", dest='use_reid', default=False, action="store_true",
                        help="For re-identification from removed or archived tracks before creating new tracks.")
    parser.add_argument('--reid_thresh', type=float, default=0.1,  # 0.2, 0.1
                        help='threshold for reid (lower will require greater similarity)')
    parser.add_argument('--window_reid', type=int, default=50, help='window of frames for the reid to be conducted.')

    # Fusion
    parser.add_argument('--fusion-method', dest='fusion_method', default='weighted_sum', type=str,
                        help='Fusion method for data association: minimum, weighted_sum, kf_gating, or hadamard '
                             '(element-wise product))')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='Value for lambda 1 - weight for IoU (for weighted_sum')
    parser.add_argument('--lambda2', type=float, default=0.1,  # 0.1 for MOT17 & MOT20, and 0.2 for DanceTrack
                        help='Value for lambda 2 - weight for Appearance (for weighted_sum')
    parser.add_argument('--lambda3', type=float, default=0.1,
                        help='Value for lambda 3 - weight for height-IoU (for weighted_sum')
    parser.add_argument('--lambda4', type=float, default=0.1,
                        help='Value for lambda 4 - weight for tracklet confidence (for weighted_sum')
    parser.add_argument('--lambda_h_kf', type=float, default=0.2,
                        help='Value for lambda_h_kf - weight for height-IoU (for kf_gating')
    parser.add_argument('--lambda_c_kf', type=float, default=0.2,
                        help='Value for lambda_c_kf - weight for tracklet confidence (for kf_gating')

    parser.add_argument('--second_matching_distance', default='iou', type=str,
                        help='Matching distance for the second matching: iou or mahalanobis')

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        if img is None:
            raise ValueError("Empty image: ", img_info["file_name"])

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

        return outputs, img_info


def image_track(predictor, vis_folder, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    if args.ablation and (args.benchmark == 'MOT17' or args.benchmark == 'MOT20'):
        files = files[len(files) // 2 + 1:]

    num_frames = len(files)

    # Tracker
    tracker = FusionSORT(args, frame_rate=args.fps)

    results = []

    for frame_id, img_path in enumerate(files, 1):

        # Detect objects
        outputs, img_info = predictor.inference(img_path, timer)
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale

            trackerTimer.tic()
            online_targets = tracker.step(detections, img_info["raw_img"])
            trackerTimer.toc()

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)  # We use the corresponding detection saved score for display purpose.

                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if args.save_frames:
            save_folder = osp.join(vis_folder, args.name)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        # Display tracks
        if args.display_tracks:
            cv2.imshow('Tracking', online_im)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        if frame_id % 20 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    res_file = osp.join(vis_folder, args.name + ".txt")

    with open(res_file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {res_file}")


def main(exp, args):

    vis_folder = os.path.join(args.output_dir, args.benchmark + '-' + args.split_to_eval, args.experiment_name, 'data')
    os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Detector Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if args.ckpt is None:
        output_dir = os.path.join(exp.output_dir, exp.exp_name)
        ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    predictor = Predictor(model, exp, args.device, args.fp16)

    image_track(predictor, vis_folder, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    data_path = args.path
    fp16 = args.fp16
    device = args.device

    if args.benchmark == 'MOT17':
        train_seqs = [2, 4, 5, 9, 10, 11, 13]
        test_seqs = [1, 3, 6, 7, 8, 12, 14]
        seqs_ext = ['FRCNN', 'DPM', 'SDP']
        MOT = 17
    elif args.benchmark == 'MOT20':
        train_seqs = [1, 2, 3, 5]
        test_seqs = [4, 6, 7, 8]
        seqs_ext = ['']
        MOT = 20
    elif args.benchmark == 'DanceTrack':
        train_seqs = [1, 2, 6, 8, 12, 15, 16, 20, 23, 24, 27, 29, 32, 33, 37, 39, 44, 45, 49, 51, 52, 53, 55, 57, 61,
                      62, 66, 68, 69, 72, 74, 75, 80, 82, 83, 86, 87, 96, 98, 99]
        val_seqs = [4, 5, 7, 10, 14, 18, 19, 25, 26, 30, 34, 35, 41, 43, 47, 58, 63, 65, 73, 77, 79, 81, 90, 94, 97]
        test_seqs = [3, 9, 11, 13, 17, 21, 22, 28, 31, 36, 38, 40, 42, 46, 48, 50, 54, 56, 59, 60, 64, 67, 70, 71, 76,
                     78, 84, 85, 88, 89, 91, 92, 93, 95, 100]
        seqs_ext = ['']
        MOT = ''
    else:
        raise ValueError("Error: Unsupported benchmark:" + args.benchmark)

    ablation = False
    if args.split_to_eval == 'train':
        seqs = train_seqs
        split = 'train'
    elif args.split_to_eval == 'val':
        seqs = train_seqs
        split = 'train'
        ablation = True
        if args.benchmark == 'MOT17':
            seqs_ext = ['FRCNN']
        if args.benchmark == 'DanceTrack':
            seqs = val_seqs
            split = 'val'
    elif args.split_to_eval == 'test':
        seqs = test_seqs
        split = 'test'
    else:
        raise ValueError("Error: Unsupported split to evaluate:" + args.split_to_eval)

    mainTimer = Timer()
    mainTimer.tic()

    for ext in seqs_ext:
        for i in seqs:
            if args.benchmark == 'MOT17' or args.benchmark == 'MOT20':
                if i < 10:
                    seq = 'MOT' + str(MOT) + '-0' + str(i)
                else:
                    seq = 'MOT' + str(MOT) + '-' + str(i)
            elif args.benchmark == 'DanceTrack':
                if i < 10:
                    seq = 'dancetrack' + '000' + str(i)
                elif (i >= 10) and (i < 100):
                    seq = 'dancetrack' + '00' + str(i)
                else:
                    seq = 'dancetrack' + '0' + str(i)
            else:
                raise ValueError("Error: Unsupported benchmark:" + args.benchmark)

            if ext != '':
                seq += '-' + ext

            args.name = seq

            args.ablation = ablation
            args.mot20 = MOT == 20
            args.fps = 30
            args.device = device
            args.fp16 = fp16
            args.batch_size = 1
            args.trt = False

            # split = 'train' if i in train_seqs else 'test'
            args.path = data_path + '/' + split + '/' + seq + '/' + 'img1'

            if args.default_parameters:

                if args.benchmark == 'MOT17':
                    if ablation:
                        args.exp_file = r'./yolox/exps/example/mot/yolox_x_ablation.py'
                        args.ckpt = r'./pretrained/bytetrack_ablation.pth.tar'  # Detection, MOT17
                    else:
                        args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_det.py'
                        args.ckpt = r'./pretrained/bytetrack_x_mot17.pth.tar'  # Detection, MOT17
                    args.fast_reid_config = r"./fast_reid/configs/MOT17/sbs_S50.yml"
                    args.fast_reid_weights = r"./pretrained/mot17_sbs_S50.pth"  # ReId, MOT17
                elif args.benchmark == 'MOT20':
                    if ablation:
                        # Just use the MOT17 test model as the ablation model for MOT20
                        args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_det.py'
                        args.ckpt = r'./pretrained/bytetrack_x_mot17.pth.tar'  # Detection, MOT17
                    else:
                        args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py'
                        args.ckpt = r'./pretrained/bytetrack_x_mot20.tar'              # Detection, MOT20
                    args.fast_reid_config = r"./fast_reid/configs/MOT20/sbs_S50.yml"
                    args.fast_reid_weights = r"./pretrained/mot20_sbs_S50.pth"      # ReId, MOT20
                    args.match_thresh = 0.7
                else:  # DanceTrack
                    # Same model for test and validation of DanceTrack
                    args.exp_file = './yolox/exps/example/mot/yolox_x_dancetrack.py'
                    args.ckpt = r'./pretrained/bytetrack_dance_model.pth.tar'    # Detection, DanceTrack
                    args.fast_reid_config = r"./fast_reid/configs/DanceTrack/sbs_S50.yml"
                    args.fast_reid_weights = r"./pretrained/dancetrack_sbs_S50.pth"      # ReId, DanceTrack

                exp = get_exp(args.exp_file, args.name)

                args.track_high_thresh = 0.6
                args.track_low_thresh = 0.1
                args.track_buffer = 30

                if seq == 'MOT17-05-FRCNN' or seq == 'MOT17-06-FRCNN':
                    args.track_buffer = 14
                elif seq == 'MOT17-13-FRCNN' or seq == 'MOT17-14-FRCNN':
                    args.track_buffer = 25
                else:
                    args.track_buffer = 30

                if seq == 'MOT17-01-FRCNN':
                    args.track_high_thresh = 0.65
                elif seq == 'MOT17-06-FRCNN':
                    args.track_high_thresh = 0.65
                elif seq == 'MOT17-12-FRCNN':
                    args.track_high_thresh = 0.7
                elif seq == 'MOT17-14-FRCNN':
                    args.track_high_thresh = 0.67
                elif seq in ['MOT20-06', 'MOT20-08']:
                    args.track_high_thresh = 0.3
                    exp.test_size = (736, 1920)

                args.new_track_thresh = args.track_high_thresh + 0.1
            else:
                exp = get_exp(args.exp_file, args.name)

            exp.test_conf = max(0.001, args.track_low_thresh - 0.01)

            # Call main function
            main(exp, args)

    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 / timer.average_time))
    print("TOTAL TIME (Tracker only): " + str(trackerTimer.total_time) + ", FPS: " + str(1.0 / trackerTimer.average_time))


