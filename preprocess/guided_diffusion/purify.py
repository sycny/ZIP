import os
import logging
import time
import glob
import sys
sys.path.append('...')
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
import torchvision.utils as tvu

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data import get_dataset, data_transform, inverse_data_transform
from preprocess.functions.ckpt_util import get_ckpt_path, download
from preprocess.functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

import torchvision.utils as tvu

from preprocess.guided_diffusion.models import Model
from preprocess.guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth
import attack.BackdoorBox as bb
from preprocess.guided_diffusion.diffusion import Diffusion

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive

class TinyImageNet_pur(ImageFolder):
    """Dataset for TinyImageNet-200-subset"""
    
    splits = ('train', 'val')
    '''
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    filename = 'tiny-imagenet-200.zip'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    ''' 
    def __init__(self, root, args, split='train', download=False, **kwargs):
        base_folder = f'../pur/{args.dataset}/{args.attack_method}/purified'
        self.data_root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", self.splits)
        
        print('fold exist or not', os.path.exists(self.split_folder))

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        super().__init__(self.split_folder, **kwargs)

    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)



class Purify(Diffusion):
    def __init__(self, args, config, type, dataset):
        super().__init__(args, config, type)
        self.poisoned_dataset = dataset
        print("The dataset for purification is:", self.poisoned_dataset)
        self.args = args
        self.type = type
        self.transform = transforms.Compose([
                              #  transforms.Resize(64),
                            # transforms.CenterCrop(64),
                             transforms.ToTensor()])
           
    def pur(self):
        
        if self.args.concat:
            
            if self.type == 'test':
                self.concat_poisoned_dataset = ConcatCLeanDataset(self.poisoned_dataset, self.args,  transform=self.transform)
            else:
                self.concat_poisoned_dataset = ConcatDataset(self.poisoned_dataset, self.args,  transform=self.transform)
        
        self.sample(self.args.simplified, self.concat_poisoned_dataset)
        
        if self.args.concat:
            
            if self.type == 'test':
                pur_dataset = SplitCLeanDataset(self.poisoned_dataset, self.args, self.type)
            else:
                pur_dataset = SplitDataset(self.poisoned_dataset, self.args, self.type)     
        else:
            pur_dataset = nonSplitDataset(self.args, self.type)

        return pur_dataset
     
#here we need the function that fits the dict label:  

class ConcatDataset(Dataset):
    def __init__(self, dataset, args, transform=None):
        self.dataset = dataset
        transformer = transforms.ToPILImage()
        self.data = lambda idx: transformer(self.dataset[idx][0])
        self.transform = transform
        self.number_in_image = (256//args.img_size)*(256//args.img_size)
        print("img_size",args.img_size)
        print("(256//args.img_size)", (256//args.img_size))
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx 
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.samples = self.dataset#.samples
        self.num_samples = len(self.samples)
        self.targets = [s[1] for s in self.samples]
        #print('self.targets',self.targets)
        self.img_paths = [s[0] for s in self.samples]
        self.concat_images = []
        self.concat_labels = []
        self.transform = transforms.Compose([
                              #  transforms.Resize(64),
                            # transforms.CenterCrop(64),
                             transforms.ToTensor()])


        for clean_class_label in self.classes:
            #print("clean_class_label", self.class_to_idx[clean_class_label])
            for pois_class_label in self.classes:
                #print("pois_class_label", self.class_to_idx[clean_class_label])
                class_indices = [i for i, t in enumerate(self.targets) if t == {'label_orig':  self.class_to_idx[clean_class_label], 'label_pois': self.class_to_idx[pois_class_label]}]
                
                num_images = len(class_indices)
                num_concat_images = math.ceil(num_images / self.number_in_image)
                last_concat_images_num = num_images % self.number_in_image

                for i in range(num_concat_images):
                    if i == num_concat_images - 1 and last_concat_images_num != 0:
                        new_image = Image.new('RGB', (256, 256))
                        row = col = 0
                        num_images_in_concat = last_concat_images_num
                    else:
                        new_image = Image.new('RGB', (256, 256))
                        row = col = 0
                        num_images_in_concat = self.number_in_image

                    for j in range(num_images_in_concat):
                        img_index = class_indices[i * self.number_in_image + j]
                        img = self.data(img_index)
                        img = img.resize((args.img_size, args.img_size))
                        new_image.paste(img, (row, col))
                        row += args.img_size
                        if row >= 252:
                            row = 0
                            col += args.img_size

                    self.concat_images.append(new_image)
                    self.concat_labels.append({'label_orig':  self.class_to_idx[clean_class_label], 'label_pois': self.class_to_idx[pois_class_label]})

        self.num_concat_images = len(self.concat_images)

    def __len__(self):
        return self.num_concat_images

    def __getitem__(self, idx):
        img = self.concat_images[idx]
        label = self.concat_labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


  
class SplitDataset(Dataset):
    def __init__(self, dataset, args, type = "train", transform=None):
        self.args = args
        self.original_dataset = dataset
        print("type",type)
        if type ==  "train":
            self.split_dir = args.splited_image_folder
        elif type ==  "test":
            self.split_dir = args.splited_test_image_folder
        elif type ==  "test_pois":
            self.split_dir = args.splited_test_image_folder_pois
        if type ==  "train":
            self.root_dir = args.image_folder
        elif type ==  "test":
            self.root_dir = args.test_image_folder
        elif type ==  "test_pois":
            self.root_dir = args.test_image_folder_pois
        self.transform = transform
        self.number_in_image = (256//args.img_size)*(256//args.img_size)
        self.number_samples = len(self.original_dataset)
        #self.targets = [s[1] for s in self.original_dataset]
        self.classes = self.original_dataset.classes
        self.class_to_idx = self.original_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        #self.img_paths = [s[0] for s in self.original_dataset]
        #self.image_files = self._get_image_files()
        self.split_images = []
        self.split_labels = []
        self.transform = transforms.Compose([
                              #  transforms.Resize(64),
                            # transforms.CenterCrop(64),
                             transforms.ToTensor()])
        
        print("split_number_in_image:", self.number_in_image)

        for clean_class_label in self.classes:
            for pois_class_label in self.classes:
                '''
                The total split image number will be the same as the original dataset img number. Therefore, let us first find out how many imgs for each class. 
                Then, in the each class, to find the split image, we need to locate the concat img, the concat img idx is equal to "split_idx//number_in_image"
                Finally, we need to locate where the img is precisely loacted in the concat img. The left boder of split image can be defined as "math.ceil(posi_concat_images/ args.img_size)", where posi_concat_images means the sequence number for the split img in the concat img.
                '''
                os.makedirs(os.path.join(self.split_dir, f"{clean_class_label}",f"{pois_class_label}"), exist_ok=True)
                
                class_path = self._get_image_files(clean_class_label, pois_class_label)
                img_slices  = []
                label_slices = []
                split_w, split_h = args.img_size, args.img_size
                
                for concat_number in range(len(class_path)):
                    img_path = class_path[concat_number]
                    concat_img = Image.open(img_path).convert('RGB')

                    for h in range(256//args.img_size):
                        for w in range(256//args.img_size):
                            img_slice = concat_img.crop((w * split_w, h * split_h, (w+1) * split_w, (h+1) * split_h))
                            exrema = img_slice.convert("L").getextrema() #check for the dark images that should not be used for test images.
                            if exrema[0]==0 and exrema[1] <= 15:
                                pass
                            else:
                                img_slice.save(os.path.join(self.split_dir, f"{clean_class_label}",f"{pois_class_label}", f"{clean_class_label}_{pois_class_label}_{concat_number}{(w,h)}.png"))
                                img_slices.append(img_slice)
                                label_slices.append({'label_orig':  self.class_to_idx[clean_class_label], 'label_pois': self.class_to_idx[pois_class_label]})
                
                self.split_images.extend(img_slices)
                self.split_labels.extend(label_slices)
            
        self.num_split_images = len(self.split_images)
        print("splited_images_number:", self.num_split_images)

             
    def _get_image_files(self, clean_class_label, pois_class_label):
            image_files = []
            class_dir = os.path.join(self.root_dir, f"{clean_class_label}", f"{pois_class_label}")
            for dirpath, _, filenames in os.walk(class_dir):
                for filename in filenames:
                    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                        image_files.append(os.path.join(dirpath, filename))
                        
            #print("concat_image_files_number:", len(image_files))
            return image_files

    def __len__(self):
        return self.num_split_images

    def __getitem__(self, idx):
        img = self.split_images[idx]
        label = self.split_labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label




#the following is working fine when "target" is in 'int' type.

class ConcatCLeanDataset(Dataset):
    def __init__(self, dataset, args, transform=None):
        print("Using ConcatCLeanDataset")
        self.dataset = dataset
        transformer = transforms.ToPILImage()
        self.data = lambda idx: transformer(self.dataset[idx][0])
        self.transform = transform
        self.number_in_image = (256//args.img_size)*(256//args.img_size)
        print("img_size",args.img_size)
        print("(256//args.img_size)", (256//args.img_size))
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.samples = self.dataset#.samples
        self.num_samples = len(self.samples)
        self.targets = [s[1] for s in self.samples]
        self.img_paths = [s[0] for s in self.samples]
        self.concat_images = []
        self.concat_labels = []
        self.transform = transforms.Compose([
                              #  transforms.Resize(64),
                            # transforms.CenterCrop(64),
                             transforms.ToTensor()])

        for class_label in self.classes:
            class_indices = [i for i, t in enumerate(self.targets) if t == self.class_to_idx[class_label]]
            num_images = len(class_indices)
            num_concat_images = math.ceil(num_images / self.number_in_image)
            last_concat_images_num = num_images % self.number_in_image

            for i in range(num_concat_images):
                if i == num_concat_images - 1 and last_concat_images_num != 0:
                    new_image = Image.new('RGB', (256, 256))
                    row = col = 0
                    num_images_in_concat = last_concat_images_num
                else:
                    new_image = Image.new('RGB', (256, 256))
                    row = col = 0
                    num_images_in_concat = self.number_in_image

                for j in range(num_images_in_concat):
                    img_index = class_indices[i * self.number_in_image + j]
                    img = self.data(img_index)
                    img = img.resize((args.img_size, args.img_size))
                    new_image.paste(img, (row, col))
                    row += args.img_size
                    if row >= 252:
                        row = 0
                        col += args.img_size

                self.concat_images.append(new_image)
                self.concat_labels.append(self.class_to_idx[class_label])

        self.num_concat_images = len(self.concat_images)

    def __len__(self):
        return self.num_concat_images

    def __getitem__(self, idx):
        img = self.concat_images[idx]
        label = self.concat_labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


  
class SplitCLeanDataset(Dataset):
    def __init__(self, dataset, args, type = "train", transform=None):
        print("Using SplitCLeanDataset")
        self.args = args
        self.original_dataset = dataset
        print("type",type)
        if type ==  "train":
            self.split_dir = args.splited_image_folder
        elif type ==  "test":
            self.split_dir = args.splited_test_image_folder
        elif type ==  "test_pois":
            self.split_dir = args.splited_test_image_folder_pois
        if type ==  "train":
            self.root_dir = args.image_folder
        elif type ==  "test":
            self.root_dir = args.test_image_folder
        elif type ==  "test_pois":
            self.root_dir = args.test_image_folder_pois
        self.transform = transform
        self.number_in_image = (256//args.img_size)*(256//args.img_size)
        self.number_samples = len(self.original_dataset)
        #self.targets = [s[1] for s in self.original_dataset]
        self.classes = self.original_dataset.classes
        self.class_to_idx = self.original_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        #self.img_paths = [s[0] for s in self.original_dataset]
        #self.image_files = self._get_image_files()
        self.split_images = []
        self.split_labels = []
        self.transform = transforms.Compose([
                              #  transforms.Resize(64),
                            # transforms.CenterCrop(64),
                             transforms.ToTensor()])
        
        print("split_number_in_image:", self.number_in_image)

        for class_label in self.classes:
            
            #The total split image number will be the same as the original dataset img number. Therefore, let us first find out how many imgs for each class. 
            #Then, in the each class, to find the split image, we need to locate the concat img, the concat img idx is equal to "split_idx//number_in_image"
            #Finally, we need to locate where the img is precisely loacted in the concat img. The left boder of split image can be defined as "math.ceil(posi_concat_images/ args.img_size)", where posi_concat_images means the sequence number for the split img in the concat img.
            
            os.makedirs(os.path.join(self.split_dir, f"{class_label}/images"), exist_ok=True)
            
            
            class_path = self._get_image_files(class_label)
            img_slices  = []
            label_slices = []
            split_w, split_h = args.img_size, args.img_size
            
            for concat_number in range(len(class_path)):
                img_path = class_path[concat_number]
                concat_img = Image.open(img_path).convert('RGB')

                for h in range(256//args.img_size):
                    for w in range(256//args.img_size):
                        img_slice = concat_img.crop((w * split_w, h * split_h, (w+1) * split_w, (h+1) * split_h))
                        exrema = img_slice.convert("L").getextrema() #check for the dark images that should not be used for test images.
                        if exrema[0]==0 and exrema[1] <= 15:
                            pass
                        else:
                            img_slice.save(os.path.join(self.split_dir, f"{class_label}/images", f"{class_label}_{concat_number}{(w,h)}.png"))
                            img_slices.append(img_slice)
                            label_slices.append(self.class_to_idx[class_label])
            
            self.split_images.extend(img_slices)
            self.split_labels.extend(label_slices)
            
        self.num_split_images = len(self.split_images)
        print("splited_images_number:", self.num_split_images)

             
    def _get_image_files(self, label):
            image_files = []
            class_dir = os.path.join(self.root_dir, f"{label}")
            for dirpath, _, filenames in os.walk(class_dir):
                for filename in filenames:
                    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                        image_files.append(os.path.join(dirpath, filename))
                        
            #print("concat_image_files_number:", len(image_files))
            return image_files

    def __len__(self):
        return self.num_split_images

    def __getitem__(self, idx):
        img = self.split_images[idx]
        label = self.split_labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label




class nonSplitDataset(Dataset):
    def __init__(self, args, type = "train"):
        
        if type ==  "train":
            self.root_dir = args.image_folder
        elif type ==  "test":
            self.root_dir = args.test_image_folder
        elif type ==  "test_pois":
            self.root_dir = args.test_image_folder_pois
        self.image_files = self._get_image_files()
        self.transform = transforms.Compose([
                              #  transforms.Resize(64),
                            # transforms.CenterCrop(64),
                             transforms.ToTensor()])
        
    def _get_image_files(self):
        image_files = []
        label = 0
        label_list = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    image_files.append(os.path.join(dirpath, filename))
                    label_list.append(label)
                label = label + 1
        return image_files, label_list
    
    def __len__(self):
            return len(self.image_files)
     
    def __getitem__(self, idx):
        
        assert(len(self.image_files[0])==len(self.image_files[1]))
        
        img_path = self.image_files[0][idx] 
        label = self.image_files[1][idx] 
        img = Image.open(img_path).convert('RGB')  
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
        


    
