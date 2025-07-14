import numpy as np
import os, sys
from datasets.base_dataset import BaseADDataset
from PIL import Image
from torchvision import transforms
from datasets.cutmix import CutMix
import random
import pandas as pd


class Fundus(BaseADDataset):

    def __init__(self, args, train = True, ref=False):
        super(Fundus).__init__()
        self.args = args
        self.train = train
        self.ref = ref
        self.classname = self.args.classname
        self.know_class = self.args.know_class
        self.pollution_rate = self.args.cont_rate
        if self.args.test_threshold == 0 and self.args.test_rate == 0:
            self.test_threshold = self.args.nAnomaly
        else:
            self.test_threshold = self.args.test_threshold

        self.transform = self.transform_train() if self.train and not self.ref else self.transform_test()
        self.transform_pseudo = self.transform_pseudo()

        self.eddfs_image_dir = './data/EDDFS/PreprocessedImages'
        self.brset_image_dir = './data/brazilian-ophthalmological/1.0.1/fundus_photos_preprocessed'

        csv_eddfs_df = pd.read_csv('./data/EDDFS/BenchReAD/train_labeled.csv')
        csv_eddfs_df_normal = csv_eddfs_df[csv_eddfs_df['abnormal'] == 0]
        csv_eddfs_df_abnormal = csv_eddfs_df[csv_eddfs_df['abnormal'] == 1]
        image_ids = list(csv_eddfs_df_normal['fnames'])
        img_paths_eddfs = [os.path.join(self.eddfs_image_dir, image_id) for image_id in image_ids]
        img_paths_eddfs_abnormal = [os.path.join(self.eddfs_image_dir, image_id) for image_id in list(csv_eddfs_df_abnormal['fnames'])]
        
        csv_brset_df = pd.read_csv('./data/brazilian-ophthalmological/1.0.1/BenchReAD/train_labeled.csv')
        csv_brset_df_normal = csv_brset_df[csv_brset_df['abnormal'] == 0]
        csv_brset_df_abnormal = csv_brset_df[csv_brset_df['abnormal'] == 1]
        image_ids = list(csv_brset_df_normal['image_id'])
        img_paths_brset = [os.path.join(self.brset_image_dir, image_id + '.jpg') for image_id in image_ids]
        img_paths_brset_abnormal = [os.path.join(self.brset_image_dir, image_id + '.jpg') for image_id in list(csv_brset_df_abnormal['image_id'])]

        normal_data = img_paths_eddfs + img_paths_brset
        abnormal_data = img_paths_eddfs_abnormal + img_paths_brset_abnormal

        self.ood_data = None

        if self.train is False:
            if self.args.test_type == 'RIADD':
                self.image_dir = './data/RIADD'
                csv_normal_df = pd.read_csv('./data/RIADD/RIADD_normal.csv')
                image_ids = list(csv_normal_df['ID'])
                image_dirs = list(csv_normal_df['dir'])
                img_paths_normal = [os.path.join(self.image_dir, image_dir, str(image_id)+'.png') for image_dir, image_id in zip(image_dirs, image_ids)]

                csv_abnormal_df = pd.read_csv('./data/RIADD/RIADD_abnormal.csv')
                image_ids = list(csv_abnormal_df['ID'])
                image_dirs = list(csv_abnormal_df['dir'])
                img_paths_abnormal = [os.path.join(self.image_dir, image_dir, str(image_id)+'.png') for image_dir, image_id in zip(image_dirs, image_ids)]

                self.images = img_paths_normal + img_paths_abnormal
                self.labels = np.array(len(img_paths_normal)*[0] + len(img_paths_abnormal)*[1])

                self.normal_idx = np.argwhere(self.labels == 0).flatten()
                self.outlier_idx = np.argwhere(self.labels == 1).flatten()

            elif self.args.test_type == 'JSIEC':
                self.image_dir = './data/JSIEC/1000images'
                csv_df = pd.read_csv(os.path.join(self.image_dir, 'JSIEC_test.csv'))
                csv_normal_df = csv_df[csv_df['abnormal'] == 0]
                csv_abnormal_df = csv_df[csv_df['abnormal'] == 1]
                image_ids = list(csv_normal_df['fnames'])
                image_dirs = list(csv_normal_df['dirs'])
                img_paths_normal = [os.path.join(self.image_dir, image_dir, image_id) for image_dir, image_id in zip(image_dirs, image_ids)]

                image_ids = list(csv_abnormal_df['fnames'])
                image_dirs = list(csv_abnormal_df['dirs'])
                img_paths_abnormal = [os.path.join(self.image_dir, image_dir, image_id) for image_dir, image_id in zip(image_dirs, image_ids)]

                self.images = img_paths_normal + img_paths_abnormal
                self.labels = np.array(len(img_paths_normal)*[0] + len(img_paths_abnormal)*[1])

                self.normal_idx = np.argwhere(self.labels == 0).flatten()
                self.outlier_idx = np.argwhere(self.labels == 1).flatten()

            elif self.args.test_type == 'valid':
                csv_eddfs_df = pd.read_csv('./data/EDDFS/BenchReAD/valid.csv')
                image_ids = list(csv_eddfs_df['fnames'])
                img_paths_eddfs = [os.path.join(self.eddfs_image_dir, image_id) for image_id in image_ids]

                csv_brset_df = pd.read_csv('./data/brazilian-ophthalmological/1.0.1/BenchReAD/valid.csv')
                image_ids = list(csv_brset_df['image_id'])
                img_paths_brset = [os.path.join(self.brset_image_dir, image_id + '.jpg') for image_id in image_ids]

                self.images = img_paths_eddfs + img_paths_brset
                self.labels = np.array(csv_eddfs_df['abnormal'].tolist() + csv_brset_df['abnormal'].tolist())
                self.normal_idx = np.argwhere(self.labels == 0).flatten()
                self.outlier_idx = np.argwhere(self.labels == 1).flatten()
            else:
                raise ValueError('Invalid test type')

        else:
            normal_label = np.zeros(len(normal_data)).tolist()
            abnormal_label = np.ones(len(abnormal_data)).tolist()

            if self.args.nAnomaly==0:
                self.images = normal_data
                self.labels = np.array(normal_label)
            else:
                self.images = normal_data + abnormal_data
                self.labels = np.array(normal_label + abnormal_label)
            self.normal_idx = np.argwhere(self.labels == 0).flatten()
            self.abnormal_idx = np.argwhere(self.labels == 1).flatten()


    def load_image(self, path):
        if 'npy' in path[-3:]:
            img = np.load(path).astype(np.uint8)
            img = img[:, :, :3]
            return Image.fromarray(img)
        return Image.open(path).convert('RGB')

    def transform_train(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size,self.args.img_size)),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def transform_pseudo(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size,self.args.img_size)),
            CutMix(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def transform_test(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rnd = random.randint(0, 1)
        if index in self.normal_idx and rnd == 0 and self.train and self.args.nAnomaly!=0:
            if self.ood_data is None:
                index = random.choice(self.normal_idx)
                image = self.load_image(self.images[index])
                transform = self.transform_pseudo
            else:
                image = self.load_image(random.choice(self.ood_data))
                transform = self.transform
            label = 2
        else:
            image = self.load_image(self.images[index])
            transform = self.transform
            label = self.labels[index]
        sample = {'image': transform(image), 'label': label}
        return sample
