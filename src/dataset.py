# -*- coding: utf-8 -*-
import os
import cv2
import torch.utils.data as data
import pandas as pd
import random
import numpy as np
from PIL import Image
import torch
import copy
from torchvision import transforms
from randaugment import RandAugment
from torch.utils.data import dataset
import pandas as pd
import glob
# from util import *

class RafDataset(data.Dataset):
    def __init__(self, args, phase, transform=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.transform = transform
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel', args.label_path), sep=' ', header=None)
        
        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
        else:
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')]
            
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        # self.aug_func = [flip_image, add_g]
        self.file_paths = []
        self.clean = (args.label_path == 'list_patition_label.txt')
        
        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(file_name)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx]) 
        image = image[:, :, ::-1] # BGR to RGB
        
        if self.transform:
            image = self.transform(image)
        # image = Image.fromarray(image)
            
        return image, label

class RafCDataset(data.Dataset):
    def __init__(self, args, phase, transform=None):
        self.rafc_path = args.rafc_path
        self.phase = phase
        self.transform = transform
        df = pd.read_csv(os.path.join(self.rafc_path, 'EmoLabel', args.label_path), sep=' ', header=None)
        
        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
        else:
            df = pd.read_csv(os.path.join(self.rafc_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')]
            
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        # self.aug_func = [flip_image, add_g]
        self.file_paths = []
        self.clean = (args.label_path == 'list_patition_label.txt')
        
        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.rafc_path, 'Image/aligned', f)
            self.file_paths.append(file_name)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx]) 
        image = image[:, :, ::-1] # BGR to RGB
        
        if self.transform:
            image = self.transform(image)
        # image = Image.fromarray(image)
            
        return image, label

class Rafunlabeled(data.Dataset):
    def __init__(self, args, phase, transform=None, strong_transform=True):
        self.raf_path = args.raf_path
        self.phase = phase
        self.strong_trasnform = strong_transform
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel', args.label_path), sep=' ', header=None)
        
        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
        else:
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')]
            
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        # self.aug_func = [flip_image, add_g]
        self.file_paths = []
        self.clean = (args.label_path == 'list_patition_label.txt')
        
        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(file_name)
            
        self.strong_transform = copy.deepcopy(transform)   
        if self.strong_transform is not None:
            self.strong_transform.transforms.insert(0, RandAugment(3, 5))
            
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # label = self.label[idx]
        image1 = cv2.imread(self.file_paths[idx]) 
        image2 = cv2.imread(self.file_paths[idx])
        image1 = image1[:, :, ::-1] # BGR to RGB
        
        if self.strong_trasnform is not None:
            image1 = self.transform(image1)
            image2 = self.strong_trasnform(image1)
        
        image2 = transforms.ToTensor()(image2)
        # image = Image.fromarray(image)
            
        return (image1, image2), idx


class LFW(data.Dataset):
    def __init__(self, args, transform=None, strong_transform=True):
        self.dataset = args.lfw_path
        # df = pd.read_csv('/home/jihyun/code/label_smec/pseudo_label/pseudo_labels_128_1_only_label.txt', sep=' ', header=None)
        self.file_list = os.listdir(self.dataset)
        self.strong_transform = strong_transform 
        self.file_path = []
        
        # self.label = df.iloc[:, 0].values
        
        for f in self.file_list:
            image_name = os.path.join(self.dataset, f)
            self.file_path.append(image_name)
        
        self.transform = transform
        if self.strong_transform:
            self.strong_transform = copy.deepcopy(transform)
            self.strong_transform.transforms.insert(1, RandAugment(3, 5))
        
        self.label = list(range(len(self.file_path)))
    
    def update_label(self, new_label): # getitem에서 idx 대신 target을 내보내도록 업데이트
        self.label = new_label
    
    def restore_label(self): # getitem에서 idx를 내보내도록 복구
        self.label = list(range(len(self.file_path)) )

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        # label = self.label[idx]
        image1 = cv2.imread(self.file_path[idx])
        image2 = cv2.imread(self.file_path[idx])
        
        # image2 = 
        # image1 = transforms.ToTensor()(image1) # 이미지를 텐서로 변환
        # image2 = transforms.ToTensor()(image2)
        
        image1 = self.transform(image1)
        if self.strong_transform is not None:
            # image2 = Image.fromarray(image2)
            image2 = self.strong_transform(image2)
        else:
            image2 = self.transform(image2)
                    
        return (image1, image2), self.label[idx], self.file_path[idx]
    
class AffectNet(data.Dataset):
    def __init__(self, args, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.aff_path = args.aff_path
        

        df = self.get_df()
        # df = self.get_df()

        self.data = df[df['phase'] == phase]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

    def get_df(self):
        train_path = os.path.join(self.aff_path,'train_set/')
        val_path = os.path.join(self.aff_path,'val_set/')
        data = []
        
        for anno in glob.glob(train_path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(train_path,f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append(['train',img_path,label])
        
        for anno in glob.glob(val_path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(val_path,f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append(['val',img_path,label])
        
        return pd.DataFrame(data = data,columns = ['phase','img_path','label'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB') # (C, H, W)
        label = self.label[idx]

        image = self.transform(image)
        
        return image, label
    
def _process_row(row):
    """
    Process a single dataframe row, returns the argmax label
    :param row:
    :return:
    """
    return np.argmax(row)

class FERPlusDataset(dataset.Dataset):
    """
    Creats a PyTorch custom Dataset for batch iteration
    """
    def __init__(self, args, mode="train", transform=None):
        self.fer_data_dir = args.ferplus_path
        self.transform = transform
        self.mode = mode
        if self.mode == "train":
            self.img_dir = os.path.join(self.fer_data_dir, "FER2013Train")
        elif self.mode == "val":
            self.img_dir = os.path.join(self.fer_data_dir, "FER2013Valid")
        elif self.mode == "test":
            self.img_dir = os.path.join(self.fer_data_dir, "FER2013Test")
        self.label_file = os.path.join(self.img_dir, "label.csv")

        self.label_data_df = pd.read_csv(self.label_file, header=None)
        self.label_data_df.columns = [
            "img_name", "dims", "0", "1", "2", "3", "4", "5", "6", "7",
            "Unknown", "NF"
        ]

        # The arg-max label is the selected as the actual label for Majority Voting
        self.label_data_df['actual_label'] = self.label_data_df[[
            '0', '1', '2', '3', '4', '5', '6', '7'
        ]].apply(lambda x: _process_row(x), axis=1)

        # get all ilocs with actual label 0
        self.label_data_df.sort_values(by=['img_name'])

        if mode == "train":
            locs0 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '0'].index.values)

            # Sampling can be turned off otherwise selects only 40% of neutral ~ 4k images
            sample_indices0 = random.Random(1).sample(locs0,
                                                      int(len(locs0) * 0.6))
            locs1 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '1'].index.values)

            # Select only 50% of neutral ~ 4k images
            sample_indices1 = random.Random(1).sample(locs1,
                                                      int(len(locs1) * 0.5))

            locs5 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '5'].index.values)
            locs6 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '6'].index.values)
            locs7 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '7'].index.values)
            self.label_data_df = self.label_data_df.drop(sample_indices0 +
                                                         sample_indices1 +
                                                         locs5 + locs6 + locs7)

        elif mode in ["val", "test"]:
            locs5 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '5'].index.values)
            locs6 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '6'].index.values)
            locs7 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '7'].index.values)
            self.label_data_df = self.label_data_df.drop(locs5 + locs6 + locs7)

        self.image_file_names = self.label_data_df['img_name'].values

    def __getitem__(self, idx):
        img_file_name = self.image_file_names[idx]
        img_file = os.path.join(self.img_dir, img_file_name)
        img = Image.open(img_file).convert('RGB')
        img_class = self.get_class(img_file_name)
        label = torch.tensor(img_class).to(torch.long) # Convert to tensor

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_class(self, file_name):
        """
        Returns the label for a corresponding file
        :param file_name: Image file name
        :return:
        """
        row_df = self.label_data_df[self.label_data_df["img_name"] ==
                                    file_name]
        init_val = -1
        init_idx = -1
        for x in range(2, 10):
            max_val = max(init_val, row_df.iloc[0].values[x])
            if max_val > init_val:
                init_val = max_val
                init_idx = int(
                    x - 2
                )  # Labels indices must start at 0, -2 if all else -4!!!!!!
        return init_idx

    def __len__(self):
        return len(self.image_file_names)
