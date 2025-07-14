import os
from enum import Enum
import numpy as np
import pandas as pd
import PIL
import torch
from torchvision import transforms
import random
random.seed(0)

_CLASSNAMES = [
    "fundus",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "valid"
    TEST = "valid"
    RIADD = "RIADD"
    JSIEC = "JSIEC"


class FundusDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Fundus.
    """

    def __init__(
        self,
        source,
        classname,
        resize=224,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.eddfs_image_dir = './data/EDDFS/PreprocessedImages'
        self.brset_image_dir = './data/brazilian-ophthalmological/1.0.1/fundus_photos_preprocessed'

        if self.split == DatasetSplit.TRAIN:
            csv_eddfs_df = pd.read_csv('./data/EDDFS/BenchReAD/train_labeled.csv')
            csv_eddfs_df = csv_eddfs_df[csv_eddfs_df['abnormal'] == 0]
            image_ids = list(csv_eddfs_df['fnames'])
            img_paths_eddfs = [os.path.join(self.eddfs_image_dir, image_id) for image_id in image_ids]

            csv_brset_df = pd.read_csv('./data/brazilian-ophthalmological/1.0.1/BenchReAD/train_labeled.csv')
            csv_brset_df = csv_brset_df[csv_brset_df['abnormal'] == 0]
            image_ids = list(csv_brset_df['image_id'])
            img_paths_brset = [os.path.join(self.brset_image_dir, image_id + '.jpg') for image_id in image_ids]

            self.img_paths = img_paths_eddfs + img_paths_brset
            random.shuffle(self.img_paths)
            
            self.targets = np.zeros(len(self.img_paths))

        elif self.split == DatasetSplit.VAL:
            csv_eddfs_df = pd.read_csv('./data/EDDFS/BenchReAD/valid.csv')
            image_ids = list(csv_eddfs_df['fnames'])
            img_paths_eddfs = [os.path.join(self.eddfs_image_dir, image_id) for image_id in image_ids]

            csv_brset_df = pd.read_csv('./data/brazilian-ophthalmological/1.0.1/BenchReAD/valid.csv')
            image_ids = list(csv_brset_df['image_id'])
            img_paths_brset = [os.path.join(self.brset_image_dir, image_id + '.jpg') for image_id in image_ids]

            self.img_paths = img_paths_eddfs + img_paths_brset
            self.targets = np.array(csv_eddfs_df['abnormal'].tolist() + csv_brset_df['abnormal'].tolist())

        elif self.split == DatasetSplit.RIADD:
            self.image_dir = './data/RIADD'
            csv_normal_df = pd.read_csv('./data/RIADD/RIADD_normal.csv')
            image_ids = list(csv_normal_df['ID'])
            image_dirs = list(csv_normal_df['dir'])
            img_paths_normal = [os.path.join(self.image_dir, image_dir, str(image_id)+'.png') for image_dir, image_id in zip(image_dirs, image_ids)]

            csv_abnormal_df = pd.read_csv('./data/RIADD/RIADD_abnormal.csv')
            image_ids = list(csv_abnormal_df['ID'])
            image_dirs = list(csv_abnormal_df['dir'])
            img_paths_abnormal = [os.path.join(self.image_dir, image_dir, str(image_id)+'.png') for image_dir, image_id in zip(image_dirs, image_ids)]
            self.img_paths = img_paths_normal + img_paths_abnormal
            self.targets = np.array(len(img_paths_normal)*[0] + len(img_paths_abnormal)*[1])

        elif self.split == DatasetSplit.JSIEC:
            self.image_dir = './data/JSIEC/1000images'
            csv_df = pd.read_csv('./data/JSIEC/1000images/JSIEC_test.csv')
            csv_normal_df = csv_df[csv_df['abnormal'] == 0]
            csv_abnormal_df = csv_df[csv_df['abnormal'] == 1]
            image_ids = list(csv_normal_df['fnames'])
            image_dirs = list(csv_normal_df['dirs'])
            img_paths_normal = [os.path.join(self.image_dir, image_dir, image_id) for image_dir, image_id in zip(image_dirs, image_ids)]

            image_ids = list(csv_abnormal_df['fnames'])
            image_dirs = list(csv_abnormal_df['dirs'])
            img_paths_abnormal = [os.path.join(self.image_dir, image_dir, image_id) for image_dir, image_id in zip(image_dirs, image_ids)]
            self.img_paths = img_paths_normal + img_paths_abnormal
            self.targets = np.array(len(img_paths_normal)*[0] + len(img_paths_abnormal)*[1])
        else:
            raise ValueError(f"Invalid split: {self.split}")

        self.labels = ["good" if x == 0 else "bad" for x in self.targets]
        print(len(self.labels))
        self.data_to_iterate = list(zip(self.img_paths, self.labels))

        self.transform_img = [
            transforms.Resize(resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees, 
                                    translate=(translate, translate),
                                    scale=(1.0-scale, 1.0+scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname = 'fundus'
        anomaly = 'good' if self.targets[idx] == 0 else 'bad'
        image_path = self.img_paths[idx]

        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        mask = torch.ones([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": image_path.split("/")[-1],
            "image_path": image_path,
        }

    def __len__(self):
        if self.split == DatasetSplit.TRAIN:
            return 50
        else:
            return len(self.img_paths)
