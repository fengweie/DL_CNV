# A Pytorch Implementation of Automated segmentation of choroidal neovascularization on optical coherence tomography angiography images of neovascular age-related macular degeneration patients based on deep learning (Journal of Big Data)

## Requirements
Prerequisites
* python3
* numpy
* pillow
* opencv-python
* scikit-learn
* tensorboardX
* visdom
* pytorch
* torchvision


## Data preparation

By default, we put the datasets in `./data/datasets/` and save trained models in `./models/` (soft link is suggested). You can set the `--data_dir` argument to `/your/train_data/path/`, the `--val_data_dir` argument to `/your/val_data/path/`, and the `--models_dir` argument to `/your/models/path/` when running all experiments below.


## Training segmentation model
In order to train the segmentation model, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python front_main.py
```

The related model parameters can be modified in `option.py`.

Please cite our paper if you find it useful for your research.

```
@article{feng2023automated,
  title={Automated segmentation of choroidal neovascularization on optical coherence tomography angiography images of neovascular age-related macular degeneration patients based on deep learning},
  author={Feng, Wei and Duan, Meihan and Wang, Bingjie and Du, Yu and Zhao, Yiran and Wang, Bin and Zhao, Lin and Ge, Zongyuan and Hu, Yuntao},
  journal={Journal of Big Data},
  volume={10},
  number={1},
  pages={111},
  year={2023},
  publisher={Springer}
}
```
