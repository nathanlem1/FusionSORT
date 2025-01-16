# FusionSORT
This code implements the paper [FusionSORT: Fusion Methods for Online Multi-object Visual Tracking](https://arxiv.org/abs/2501.00843).

## Abstract
In this work, we investigate four different fusion methods for associating detections to tracklets in multi-object 
visual tracking. In addition to considering strong cues such as motion and appearance information, we also consider 
weak cues such as height intersection-over-union (height-IoU) and tracklet confidence information in the data 
association using different fusion methods. These fusion methods include minimum, weighted sum based on IoU, 
Kalman filter (KF) gating, and hadamard product of costs due to the different cues. We conduct extensive evaluations 
on validation sets of MOT17, MOT20 and DanceTrack datasets, and find out that the choice of a fusion method is key for 
data association in multi-object visual tracking. We hope that this investigative work helps the computer vision 
research community to use the right fusion method for data association in multi-object visual tracking.

## Installation

**Step 1.** Install torch and matched torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).
The code was tested using torch 2.2.2+cu118 and torchvision 0.17.2+cu118. 

**Step 2.** Clone FusionSORT and install it. Check if yours is pip or pip3 and python or python3.
```shell
git clone https://github.com/nathanlem1/FusionSORT.git
cd FusionSORT/
pip install -r requirements.txt
python setup.py develop  
```
Please check the [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and
[FastReID](https://github.com/JDAI-CV/fast-reid) READMEs for any installation issues.

**Step 3.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

**Step 4.** Others
```shell
# Cython-bbox
pip install cython_bbox

# faiss cpu / gpu
pip install faiss-cpu
pip install faiss-gpu
```

## Data Preparation

Download [MOT17](https://motchallenge.net/data/MOT17/), [MOT20](https://motchallenge.net/data/MOT20/) and 
[DanceTrack](https://github.com/DanceTrack/DanceTrack). And put them in the `FusionSORT/datasets` folder according to 
the following structure:

```
datasets
|——————MOT17
|        └——————train
|        └——————test
|——————MOT20
|        └——————train
|        └——————test
|——————DanceTrack
|        └——————train
|        └——————test
|        └——————val
```

## Model Zoo
Download the weights of the trained object detector and appearance (ReID) models to the `FusionSORT/pretrained` 
directory (do NOT untar the `.pth.tar` YOLOX files) using the following links.

- For MOT17 and MOT20 datasets, we used the publicly available YOLOX object detector models trained by [ByteTrack](https://github.com/ifzhang/ByteTrack). They can be downloaded from [bytetrack_x_mot17](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing) (test model), [bytetrack_x_mot20](https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view?usp=sharing) (test model) and [bytetrack_ablation](https://drive.google.com/file/d/1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob/view?usp=sharing) (validation model for MOT17).

- For MOT17 and MOT20 datasets, we used the publicly available appearance (ReID) models trained by [BoT-SORT](https://github.com/NirAharon/BoT-SORT). They can be downloaded from [MOT17-SBS-S50](https://drive.google.com/file/d/1QZFWpoa80rqo7O-HXmlss8J8CnS7IUsN/view?usp=sharing) and [MOT20-SBS-S50](https://drive.google.com/file/d/1KqPQyj6MFyftliBHEIER7m_OrGpcrJwi/view?usp=sharing).

- For DanceTrack dataset, we used the publicly available YOLOX object detector and appearance (ReID) models trained by [HybridSORT](https://github.com/ymzis69/HybridSORT).  They can be downloaded from [bytetrack_dance_model](https://drive.google.com/file/d/1b1b6CyA01HHSAXRDJPoRDTDz715JRmHt/view?usp=sharing) (test model) and [dancetrack_sbs_S50](https://drive.google.com/file/d/1KFjYc7B2Iv7MYPKBWxePZ1bZOdv6K-RF/view?usp=sharing), respectively.

Note that we use MOT17 test detector model as the validation model for MOT20 dataset. In addition, we use the DanceTrack test detector model as a validation model, in our experiments, similar to [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT).


## Tracking 
* **Evaluation on validation set**

To evaluate on MOT17 val dataset, you need to run:

```shell
# Using motion only
python run_fusion_sort.py --path ./datasets/MOT17 --default-parameters --benchmark MOT17 --eval val --experiment-name FSORT1 --fp16 --fuse

# Using motion and appearance
python run_fusion_sort.py --path ./datasets/MOT17 --default-parameters --with-appearance --fusion-method weighted_sum --benchmark MOT17 --eval val --experiment-name FSORT1 --fp16 --fuse

# Using motion, appearance, hiou and confidence distances
python run_fusion_sort.py --path ./datasets/MOT17 --default-parameters --with-appearance --with-hiou --with-confidence --fusion-method weighted_sum --benchmark MOT17 --eval val --experiment-name FSORT1 --fp16 --fuse
``` 
You only need to change the dataset type to 'MOT20' or 'DanceTrack' to run the tracker on either of them. Note 
that MOT17 and MOT20 validation sets are the second half of the train sets since they do not have separate 
validation sets as the DanceTrack dataset, which has a separate validation set. You can use the fusion method you want: 
minimum, weighted_sum, kf_gating or hadamard.

* **Evaluation on test set**

To evaluate on MOT17 test dataset, you need to run:

```shell
# Using motion only
python run_fusion_sort.py --path ./datasets/MOT17 --default-parameters --benchmark MOT17 --eval test --experiment-name FSORT1 --fp16 --fuse

# Using motion and appearance
python run_fusion_sort.py --path ./datasets/MOT17 --default-parameters --with-appearance --fusion-method weighted_sum --benchmark MOT17 --eval test --experiment-name FSORT1 --fp16 --fuse

# Using motion, appearance, hiou and confidence distances
python run_fusion_sort.py --path ./datasets/MOT17 --default-parameters --with-appearance --with-hiou --with-confidence --fusion-method weighted_sum --benchmark MOT17 --eval test --experiment-name FSORT1 --fp16 --fuse
```
You only need to change the dataset type to 'MOT20' or 'DanceTrack' to run the tracker on either of them. You can use the 
fusion method you want: minimum, weighted_sum, kf_gating or hadamard.


* **Interpolation**

This is optional. We didn't apply any interpolation for our experimental results reported in the paper.

For using linear interpolation, run the following code:
```shell
python tools/linear_interpolation.py --txt_path <path_to_track_result>
```

You can also use Gaussian-smoothed interpolation (GSI) as in [StrongSORT](https://github.com/dyhBUPT/StrongSORT).


## Evaluation

You can use the official MOTChallenge evaluation code from [TrackEval](https://github.com/JonathonLuiten/TrackEval) 
to evaluate the MOT17, MOT20  and DanceTrack `train` or `val` datasets. For our experimental analysis, we use `val`. 
To evaluate on `val` datasets, you need to use `val` for `--SPLIT_TO_EVAL`. Hence, to evaluate on MOT17, MOT20 or 
DanceTrack datasets, you can run the following code on terminal:

```shell
python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL val --TRACKERS_TO_EVAL FSORT1 --METRICS HOTA CLEAR Identity VACE --GT_FOLDER results/gt/ --TRACKERS_FOLDER results/trackers/ --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```
Please note the output folder name `FSORT1`.


## Demo
FusionSORT demo based on YOLOX-X detector for three different demo types: webcam, image and video.

```shell
# Using webcam
python tools/demo_fusionsort.py --demo_type webcam --camid 0 --with-appearance --fusion-method weighted_sum --experiment-name FSORT1 --fp16 --fuse --display_tracks --save_result

# Using image
python tools/demo_fusionsort.py --demo_type image --path <path_to_images> --with-appearance --fusion-method weighted_sum --experiment-name FSORT1 --fp16 --fuse --display_tracks --save_result

# Using video
python tools/demo_fusionsort.py --demo_type video --path <path_to_video> --with-appearance --fusion-method weighted_sum --experiment-name FSORT1 --fp16 --fuse --display_tracks --save_result
```
You can use the fusion method you want: minimum, weighted_sum, kf_gating or hadamard.


## Citation
If you find this work useful, please consider citing our paper:
```
@misc{baisa2025fusionsortfusionmethodsonline,
      title={FusionSORT: Fusion Methods for Online Multi-object Visual Tracking}, 
      author={Nathanael L. Baisa},
      year={2025},
      eprint={2501.00843},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.00843}, 
}
```


## Acknowledgement

A significant part of the codes and ideas are borrowed from 
[BoT-SORT](https://github.com/NirAharon/BoT-SORT),
[ByteTrack](https://github.com/ifzhang/ByteTrack),
[HybridSORT](https://github.com/ymzis69/HybridSORT),
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and
[FastReID](https://github.com/JDAI-CV/fast-reid).
Thanks for their excellent work!
