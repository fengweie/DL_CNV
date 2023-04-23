# A Pytorch Implementation of Automated segmentation of choroidal neovascularization on optical coherence tomography angiography images of neovascular age-related macular degeneration patients based on deep learning (Journal of Big Data)

## Requirements
`pip3 install -r requirements.txt`


## Data preparation

By default, we put the datasets in `./data/datasets/` and save trained models in `./data/experiments/` (soft link is suggested). You may also use any other directories you like by setting the `--dataset_root` argument to `/your/data/path/`, and the `--exp_root` argument to `/your/experiment/path/` when running all experiments below.

- For CIFAR-10, CIFAR-100, and SVHN, simply download the datasets and put into `./data/datasets/`.
- For OmniGlot, after downloading, you need to put `Alphabet_of_the_Magi, Japanese_(katakana), Latin, Cyrillic, Grantha` from  `imags_background` folder into `images_background_val` folder, and put the rest alphabets into `images_background_train` folder.
- For ImageNet, we provide the exact split files used in the experiments following existing work. To download the split files, run the command:
``
sh scripts/download_imagenet_splits.sh
``
. The ImageNet dataset folder is organized in the following way:

    ```
    ImageNet/imagenet_rand118 #downloaded by the above command
    ImageNet/images/train #standard ImageNet training split
    ImageNet/images/val #standard ImageNet validation split
    ```

## Training segmentation model

```
CUDA_VISIBLE_DEVICES=0 python selfsupervised_learning.py --dataset_name cifar10 --model_name rotnet_cifar10 --dataset_root ./data/datasets/CIFAR/
```

``--dataset_name`` can be one of ``{cifar10, cifar100, svhn}``; ``--dataset_root`` is set to ``./data/datasets/CIFAR/`` for CIFAR10/CIFAR100 and ``./data/datasets/SVHN/`` for SVHN.

Our code for step 1 is based on the official code of the [RotNet paper](https://arxiv.org/pdf/1803.07728.pdf).


<!-- ## Contact
- lyhaolive@gmail.com --> -->
