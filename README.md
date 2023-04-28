# A Pytorch Implementation of Automated segmentation of choroidal neovascularization on optical coherence tomography angiography images of neovascular age-related macular degeneration patients based on deep learning (Journal of Big Data)

## Requirements
`pip3 install -r requirements.txt`


## Data preparation

By default, we put the datasets in `./data/datasets/` and save trained models in `./models/` (soft link is suggested). You can set the `--data_dir` argument to `/your/train_data/path/`, the `--val_data_dir` argument to `/your/val_data/path/`, and the `--models_dir` argument to `/your/models/path/` when running all experiments below.


## Training segmentation model
In order to train the segmentation model, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python front_main.py
```

The related model parameters can be modified in `option.py`.
