# 导入库
import os

import numpy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.pytorch.functional import img_to_tensor
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import io

class mydatasets(torch.utils.data.Dataset):
    def __init__(self,images_dir,masks_dir,cmap,labels,val=False):
        self.image_dir = images_dir
        self.mask_dir = masks_dir
        self.cmap = cmap
        self.class_num = list(range(len(labels)))
        self.train_transform = A.Compose([
            A.CLAHE(),
            A.Resize(320, 320),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.GaussianBlur(blur_limit=(1, 5))
        ])
        self.val_transform = A.Compose([
            A.Resize(480,480)
        ])
        self.val = val
        # self.ids = os.listdir(images_dir)
        # self.images_fps = [os.path.join(images_dir,image_id) for image_id in self.ids]
        # self.masks_fps = [os.path.join(masks_dir,image_id) for image_id in self.ids]

    def __getitem__(self, index):
        image = self.image_dir[index]
        seg = self.mask_dir[index]

        image = io.imread(str(image))
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        seg = io.imread(str(seg))
        # print(seg)
        seg = cv.cvtColor(seg,cv.COLOR_BGR2RGB)
        # print(seg.shape)
        # masks = [(seg==v) for v in self.class_num]
        # mask = np.stack(masks,axis=-1).astype('float')

        if self.val:
            aug = self.val_transform(image=image,mask=seg)
        else:
            aug = self.train_transform(image=image,mask=seg)
        image,seg = aug['image'],aug['mask']
        # print(seg.squeeze().shape)
        mask_shape = *seg.shape[:2],len(self.cmap)
        # print(mask_shape.shape)
        mask_z = np.zeros(mask_shape)
        for i, color in self.cmap.items():
            # print(color)
            temp = np.equal(seg,color)
            mask_z[:,:,i] = np.all(temp,axis=-1)
        # mask_z = np.stack(mask_z,axis=-1).astype(np.float32)
        mask = torch.tensor(mask_z,dtype=torch.float32).permute(2,0,1)

        image = img_to_tensor(image)
        # print(image.shape,mask.shape)
        return image,mask

    def __len__(self):
        # assert len(self.image_dir) == len(self.mask_dir)
        # print(len(self.image_dir))
        return len(self.image_dir)


#root path
Cam_root_path = r'D:\Test_data\VIT\image\CamVid\CamVid'

#read csv-file
classes = pd.read_csv(Cam_root_path +'/' + 'class_dict.csv')
labels = dict(zip(range(len(classes)),classes['name'].tolist()))
classes_values = [i for i in labels.values()]
color_map = dict(zip(range(len(classes)),classes.iloc[:,1:].values.tolist()))
palette = [color for color in color_map.values()]

#test path
test_image_path = Cam_root_path + '/' + os.listdir(Cam_root_path)[1]
test_label_path = Cam_root_path + '/' + os.listdir(Cam_root_path)[2]
test_images = [os.path.join(test_image_path,image_id) for image_id in os.listdir(test_image_path)]
test_labels = [os.path.join(test_label_path,label_id) for label_id in os.listdir(test_label_path)]

#train path
train_image_path = Cam_root_path + '/' + os.listdir(Cam_root_path)[3]
train_label_path = Cam_root_path + '/' + os.listdir(Cam_root_path)[4]
train_images = [os.path.join(train_image_path,image_id) for image_id in os.listdir(train_image_path)]
train_labels = [os.path.join(train_label_path,label_id) for label_id in os.listdir(train_label_path)]

#val path
val_image_path = Cam_root_path + '/' + os.listdir(Cam_root_path)[5]
val_label_path = Cam_root_path + '/' + os.listdir(Cam_root_path)[6]
val_images = [os.path.join(val_image_path,image_id) for image_id in os.listdir(val_image_path)]
val_labels = [os.path.join(val_label_path,label_id) for label_id in os.listdir(val_label_path)]

labelpath = [test_label_path,train_label_path,val_label_path]
labellist = [test_labels,train_labels,val_labels]
#spilt dataset
train_dataset = mydatasets(train_images,train_labels,cmap=color_map,labels=labels)
test_dataset = mydatasets(test_images,test_labels,cmap=color_map,labels=labels)
val_dataset = mydatasets(val_images,val_labels,cmap=color_map,labels=labels)

# train_dataset = CamVidDataset(train_images,train_labels)
# test_dataset = CamVidDataset(test_images,test_labels)
# val_dataset = CamVidDataset(val_images,val_labels)

#spilt loader
train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=4,pin_memory=True)
test_loader = DataLoader(test_dataset,batch_size=8,shuffle=True,num_workers=4,pin_memory=True)
val_loader = DataLoader(val_dataset,batch_size=8,shuffle=True,num_workers=4,pin_memory=True)


Cam_vid_datasets = {'train':train_loader,
                    'test':test_loader,
                    'val':val_loader}

if __name__ == '__main__':
    # print(Cam_vid_datasets)
    # print(classes_values)
    # print(color_map)
    # for i,color in color_map.items():
    #     print(i,color)
    # print(test_images)
    # print(test_labels)
    # print(train_images)
    # print(train_labels)
    # print(val_images)
    # print(val_labels)
    # print(palette)
    img,mask = next(iter(val_loader))
    print(img.shape,mask.shape)
    # plt.imshow(img[0].permute(1,2,0))
    # print(img.shape, mask.shape)
    # n = np.argmax(mask.numpy(),axis=1)
    # print(n.shape)
    # print(n.dtype)
    # print(np.unique(n),len(np.unique(n)))
    # plt.subplot(141)
    # plt.imshow(n[0])
    # plt.subplot(142)
    # plt.imshow(mask[0,0,:,:])
    # plt.subplot(143)
    # plt.imshow(mask[0, 1, :, :])
    # plt.subplot(144)
    # plt.imshow(mask[0, 2, :, :])
    # plt.show()
