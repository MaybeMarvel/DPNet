# 导入库
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
import albumentations as A
import matplotlib.pyplot as plt
import cv2
import glob
from sklearn.model_selection import train_test_split


def split_ids(len_ids):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices


class Kvasir_Dataset(Dataset):
    def __init__(self,images_dir,masks_dir,train):
        self.images = images_dir
        self.gts = masks_dir
        self.train = train
        self.cmap = [[0],[255]]

        self.to_transform = A.Compose([A.CLAHE(),
                                       A.OneOf([ A.HorizontalFlip(p=0.5),
                                       A.VerticalFlip(p=0.5),
                                       A.ShiftScaleRotate()],p=0.5)])
        self.train_transform = A.Compose([
            A.Resize(352, 352,interpolation=0),
            A.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
            A.GaussianBlur((25, 25), sigma_limit=(0.001, 2.0)),
            A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01)
        ])
        self.gt_transform = A.Compose([
            A.Resize(352, 352,interpolation=0)])

    def __getitem__(self, index):
        image = self.images[index]
        image = cv2.imread(image)
        gt = self.gts[index]
        gt = cv2.imread(gt)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # if self.mode == 'train':
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY)

        if self.train:
            TR = self.to_transform(image=image,mask=gt)
            image,gt = TR['image'],TR['mask']

        aug = self.train_transform(image=image)
        image = aug['image']
        image = torch.tensor(image,dtype=torch.float32).permute(2,0,1)
        gt = self.gt_transform(image=gt)
        gt = gt['image']
        ret,gt=cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
        # n = np.unique(gt)
        gt = np.expand_dims(gt,axis=2)
        gt = mask2onehot(gt,self.cmap)
        gt = gt.transpose([2,0,1])
        # print(gt.shape)
        gt = torch.tensor(gt,dtype=torch.float32)

        return image, gt

    def __len__(self):
        return len(self.images)

class SkinDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images = images_dir
        self.gts = masks_dir
        self.cmap = [[0],[255]]
        self.size = len(self.images)

        self.train_transform = A.Compose([
            A.Resize(352, 352, interpolation=0),
            A.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
            A.GaussianBlur((25, 25), sigma_limit=(0.001, 2.0)),
            A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01)
        ])
        self.gt_transform = A.Compose([
            A.Resize(352, 352, interpolation=0)])


    def __getitem__(self, index):
        image = self.images[index]
        image = cv2.imread(image)
        gt = self.gts[index]
        gt = cv2.imread(gt)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if self.mode == 'train':
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

        aug = self.train_transform(image=image)
        image = aug['image']
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        gt = self.gt_transform(image=gt)
        gt = gt['image']
        ret, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
        # n = np.unique(gt)
        gt = np.expand_dims(gt, axis=2)
        gt = mask2onehot(gt, self.cmap)
        gt = gt.transpose([2, 0, 1])
        # print(gt.shape)
        gt = torch.tensor(gt, dtype=torch.float32)

        return image, gt

    def __len__(self):
        return len(self.images)


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


def get_dataloaders(input_paths, target_paths, batch_size,dataset):
    if dataset == "Kvasir" or "CVC":
        train_dataset = Kvasir_Dataset(
            images_dir=input_paths,
            masks_dir=target_paths,
            train = True
        )

        test_dataset = Kvasir_Dataset(
            images_dir=input_paths,
            masks_dir=target_paths,
            train = False,
        )

        val_dataset = Kvasir_Dataset(
            images_dir=input_paths,
            masks_dir=target_paths,
            train=False,
        )
    elif dataset == "HAM":
        train_dataset = SkinDataset(
            images_dir=input_paths,
            masks_dir=target_paths,
        )

        test_dataset = SkinDataset(
            images_dir=input_paths,
            masks_dir=target_paths,
        )

        val_dataset = SkinDataset(
            images_dir=input_paths,
            masks_dir=target_paths,
        )

    train_indices, test_indices, val_indices = split_ids(len(input_paths))

    train_dataset = data.Subset(train_dataset, train_indices)
    val_dataset = data.Subset(val_dataset, val_indices)
    test_dataset = data.Subset(test_dataset, test_indices)

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    return train_dataloader, test_dataloader, val_dataloader




if __name__ == '__main__':
    root = os.path.join(os.path.dirname(__file__),r'../../image/CVC-ClinicDB/CVC-ClinicDB/')
    img_path = root + "Original/*"
    input_paths = sorted(glob.glob(img_path))
    depth_path = root + "Ground Truth/*"
    target_paths = sorted(glob.glob(depth_path))
    train_dataloader, _, val_dataloader = get_dataloaders(
        input_paths, target_paths, batch_size=4,dataset="HAM"
    )

    img,gt = next(iter(val_dataloader))
    print(len(val_dataloader))
    plt.subplot(121)
    plt.imshow(img[0].permute(1, 2, 0))
    plt.subplot(122)
    plt.imshow(gt[0,1,:,:],cmap='gray')
    plt.show()


