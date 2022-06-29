# I M Avatar: Implicit Morphable Head Avatars from Videos
## [Paper](https://arxiv.org/abs/2112.07471) | [Video Youtube](https://youtu.be/915baJNX-IU) | [Video Download](https://dataset.ait.ethz.ch/downloads/imaOsdfvRe/output.mp4) | [Project Page](https://ait.ethz.ch/projects/2022/IMavatar/)


<img src="assets/imavatar_real.gif" width="400" height="200"/> <img src="assets/makehuman.gif" width="400" height="200"/> 

Official Repository for CVPR 2022 paper [*I M Avatar: Implicit Morphable Head Avatars from Videos*](https://arxiv.org/abs/2112.07471). 

## Getting Started
* Clone this repo: `git clone --recursive git@github.com:zhengyuf/IMavatar.git`
* Create a conda environment `conda env create -f environment.yml` and activate `conda activate IMavatar` 
* We use `libmise` to extract 3D meshes, build `libmise` by running `cd code; python setup.py install`
* Download [FLAME model](https://flame.is.tue.mpg.de/download.php), choose **FLAME 2020** and unzip it, copy 'generic_model.pkl' into `./code/flame/FLAME2020`
* When choosing your GPU, avoid RTX30xx since it seems unstable with broyden's method, see [here](https://github.com/xuchen-ethz/snarf/issues/3#issue-1096847424) if you want to know more.
## Preparing dataset
Download a preprocessed dataset from [Google drive](https://drive.google.com/file/d/1Hzv41ZkpMK1X9h9Z-B54S-Nn1GcMveb8/view?usp=sharing) or [ETH Zurich server](https://dataset.ait.ethz.ch/downloads/IMavatar_data/data/yufeng.zip). You can run `download_data.bash`.

Or prepare your own dataset following intructions in `./preprocess/README.md`.

Link the dataset folder to `./data/datasets`. Link the experiment output folder to `./data/experiments`.

## Training
```
python scripts/exp_runner.py ---conf ./confs/IMavatar_supervised.conf [--wandb_workspace IMavatar] [--is_continue]
```
## Evaluation
Set the *is_eval* flag for evaluation, optionally set *checkpoint* (if not, the latest checkpoint will be used) and *load_path* 
```
python scripts/exp_runner.py --conf ./confs/IMavatar_supervised.conf --is_eval [--checkpoint 60] [--load_path ...]
```
## Pre-trained model
Download a pretrained model from [Google drive](https://drive.google.com/file/d/1ZaznButY_zszllbBUoF89D3gBSMn-Tcc/view?usp=sharing) or [ETH Zurich server](https://dataset.ait.ethz.ch/downloads/IMavatar_data/checkpoint/yufeng.zip). See `download_data.bash`.

## Additional features
The following features are not used in the main paper, but helpful for training.
* **Semantic-guided Training**:
set `loss.gt_w_seg` to `True` to use semantic segmentation during training. Using semantic maps leads to improved training stability, and better teeth reconstruction quality.
* **Ghost Bone**:
If FLAME global rotations in your dataset are not identity matrices, set `deformer_network.ghostbone` to `True`. This allow the shoulder and upper body to remain un-transformed.
* **Pose Optimization**:
When the FLAME parameters are noisy, I find it helpful to set `optimize_camera` to `True`. This optimizes both the FLAME pose parameters and the camera translation parameters. Similarly, set `optimize_expression` and `optimize_latent_code` to `True` to optimize input expression parameters and per-frame latent codes.

## Warning
* Our preprocessing script scales FLAME head meshes by 4 so that it would fit the unit sphere tighter. Remember to adjust camera positions accordingly if you are using your own preprocessing pipeline. 
* Multi-GPU training is not tested. We found a single GPU to be sufficient in terms of batch size.

## Citation
If you find our code or paper useful, please cite as:
```
@inproceedings{zheng2022imavatar,
  title={{I} {M} {Avatar}: Implicit Morphable Head Avatars from Videos},
  author={Zheng, Yufeng and Abrevaya, Victoria Fernández and Bühler, Marcel C. and Chen, Xu and Black, Michael J. and Hilliges, Otmar},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}
```
