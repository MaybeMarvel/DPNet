import os
import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from glob import glob
from shutil import copy2
from skimage import io
import albumentations as A
import cv2

class Skin_HAM(object):
    def __init__(self,root):
        self.root = root
        self.split_name = ['train','val','test']
        self.mask_name = ['train_mask','val_mask','test_mask']
        self.split_factor = [0.7,0.2,0.1]
        # self.images_path = root + '\images'
        # self.masks_path = root + '\masks'

    def split_set(self):
        images_path = self.root + '\images'
        masks_path = self.root + '\masks'
        images_list = glob(images_path + r'/*.jpg')
        masks_list = glob(masks_path + r'/*.png')

        for split,mask in zip(self.split_name,self.mask_name):
            split_path = os.path.join(self.root,split)
            mask_path = os.path.join(self.root, mask)
            if os.path.exists(split_path):
                pass
            else:
                os.makedirs(split_path)
            if os.path.exists(mask_path):
                pass
            else:
                os.makedirs(mask_path)
        train_size = len(images_list) * self.split_factor[0]
        val_size = len(images_list) * self.split_factor[1] + train_size
        test_size = len(images_list) * self.split_factor[2] + val_size
        for i in range(len(images_list)):
            if (i+1) <= train_size:
                copy2(images_list[i],self.root + r'\train')
                copy2(masks_list[i],self.root + r'\train_mask')
            if ((i+1) > train_size) and ((i+1) <= val_size):
                copy2(images_list[i],self.root + r'\val')
                copy2(masks_list[i], self.root + r'\val_mask')
            if ((i+1) > val_size) and ((i+1) <= test_size):
                copy2(images_list[i],self.root + r'\test')
                copy2(masks_list[i], self.root + r'\test_mask')


class SkinDataset(Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [os.path.join(image_root, images_name) for images_name in os.listdir(image_root)]
        self.gts = [os.path.join(gt_root, masks_name) for masks_name in os.listdir(gt_root)]
        # print(self.images[0])
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.cmap = [[0],[255]]
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = A.Compose([
            A.Resize(self.trainsize, self.trainsize),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.gt_transform = A.Compose([
            A.Resize(self.trainsize, self.trainsize)])

    def __getitem__(self, index):
        image = self.images[index]
        image = io.imread(image)
        gt = self.gts[index]
        gt = io.imread(gt)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        aug = self.img_transform(image=image)
        image = aug['image']
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        gt = self.gt_transform(image=gt)
        gt = gt['image']
        # ret, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
        # n = np.unique(gt)
        gt = np.expand_dims(gt, axis=2)
        gt = mask2onehot(gt, self.cmap)
        gt = gt.transpose([2, 0, 1])
        gt = torch.tensor(gt, dtype=torch.float32)

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('1')
            # return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def mask2onehot(mask,palette):
    semantic_map = []
    for color in palette:
        equality = np.equal(mask,color)
        class_map = np.all(equality,axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map,axis=-1).astype(np.float32)
    return semantic_map

def onehot2mask(mask,palette):
    x = np.argmax(mask,axis=-1)
    color_code = palette
    x = color_code[x.astype(np.uint8)]
    return x

# def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
#
#     dataset = SkinDataset(image_root, gt_root, trainsize)
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batchsize,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory)
#     return data_loader
os.path.join(os.path.dirname(__file__),r'../../image/HAM/HAM10000/val')
image_path = {'train': os.path.join(os.path.dirname(__file__),r'../../image/HAM/HAM10000/val'),
            'val': os.path.join(os.path.dirname(__file__),r'../../image/HAM/HAM10000/test')}
mask_path = {'train_mask': os.path.join(os.path.dirname(__file__),r'../../image/HAM/HAM10000/val_mask'),
            'val_mask': os.path.join(os.path.dirname(__file__),r'../../image/HAM/HAM10000/test_mask')}

train_dataset = SkinDataset(image_root=image_path['train'], gt_root=mask_path['train_mask'],
                                   trainsize=256)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=8,pin_memory=True)

# test_dataset = SkinDataset(image_root=image_path['test'], gt_root=mask_path['test_mask'],
#                                    trainsize=256)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True,num_workers=8,pin_memory=True)

val_dataset = SkinDataset(image_root=image_path['val'], gt_root=mask_path['val_mask'],
                                   trainsize=256)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True,num_workers=8,pin_memory=True)

Skin_HAM_datasets = {'train':train_dataloader,
                    'val':val_dataloader}

# Skin_HAM = Skin_HAM(root=f'D:\Test_data\VIT\image\HAM\HAM10000')
# Skin_HAM.split_set()
if __name__ == '__main__':
    # print(getloader(8,16))
    train_loader, val_loader = Skin_HAM_datasets['train'], Skin_HAM_datasets['val']
    img, mask = next(iter(val_loader))
    mask_s = mask.numpy()
    n = np.argmax(mask.numpy(),axis=1)
    # n = n*255
    print(img.shape, mask.shape)
    plt.subplot(121)
    plt.imshow(img[0].permute(1, 2, 0))
    plt.subplot(122)
    plt.imshow(n[0], cmap='gray')
    plt.show()
    # dataset = Kvasir_SEG(root=r'D:\Test_data\VIT\image\kvasir-seg\Kvasir-SEG')
    # dataset.split_set()