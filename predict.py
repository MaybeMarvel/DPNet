import os
import numpy as np
import torch
from torch import nn
from data.MedicalDataset import get_dataloaders,split_ids,Kvasir_Dataset
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from model.DPNet import DPNet
from model.ablation_model.unet import UNet
from model.ablation_model.unetpp import UnetPlusPlus
from model.ablation_model.segnet import SegNet
from utils.utils import Evaluator
from utils.metrics import dice_coeff,dice_score
from skimage import io
import cv2
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
from utils.loss import structure_loss
import time
import glob
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument(
        "--train-dataset", type=str, required=True, choices=["Kvasir", "CVC", "HAM"],dest='train_dataset'
    )
    parser.add_argument(
        "--test-dataset", type=str, required=True, choices=["Kvasir", "CVC", "HAM"],dest='test_dataset'
    )
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--model_name", type=str, default= 'DPNet', dest='model_name')

    return parser.parse_args()


class Predict(object):
    def __init__(self,args):
        super(Predict, self).__init__()
        self.args = args
        self.train_dataset_name = args.train_dataset
        self.test_dataset_name = args.test_dataset
        self.model_name = args.model_name
        self.checkpoint = self.load_model()
        self.show_score(self.checkpoint)
        self.device, self.test_dataloader, self.evaluator, self.model, self.target_paths = self.build(args)

    def load_model(self):
        if self.train_dataset_name in ['CVC','Kvasir','HAM']:
            path = os.path.join(os.path.dirname(__file__),
                                r'train_para/checkpoints/%s/%s.pth.tar' % (self.train_dataset_name,self.train_dataset_name + '_' + self.model_name))
            checkpoint = torch.load(path)
        return checkpoint



    def build(self,args):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        if args.test_dataset == args.train_dataset:
            img_path = args.root + "images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "masks/*"
            target_paths = sorted(glob.glob(depth_path))

            test_dataset = Kvasir_Dataset(
                images_dir=input_paths,
                masks_dir=target_paths,
                train=False,
            )
            # test_indices = sorted([int(os.path.basename(i)[:-4]) for i in target_paths])
            # print(test_indices,len(test_indices))
            test_dataloader = data.DataLoader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
            )
            # if args.test_dataset in ["Kvasir", "HAM"]:
            #     img_path = args.root + "images/*"
            #     input_paths = sorted(glob.glob(img_path))
            #     depth_path = args.root + "masks/*"
            #     target_paths = sorted(glob.glob(depth_path))
            # elif args.test_dataset == "CVC":
            #     img_path = args.root + "Original/*"
            #     input_paths = sorted(glob.glob(img_path))
            #     depth_path = args.root + "Ground Truth/*"
            #     target_paths = sorted(glob.glob(depth_path))
            # _, test_dataloader, _ = get_dataloaders(
            #     input_paths, target_paths, batch_size=1,dataset=args.test_dataset
            # )
            #
            # _, test_indices, _ = split_ids(len(target_paths))
            # target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]
        elif args.test_dataset != args.train_dataset:
            img_path = args.root + "images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "masks/*"
            target_paths = sorted(glob.glob(depth_path))

            test_dataset = Kvasir_Dataset(
                images_dir=input_paths,
                masks_dir=target_paths,
                train=False,
            )
            # test_indices = sorted([int(os.path.basename(i)[:-4]) for i in target_paths])
            # print(test_indices,len(test_indices))
            test_dataloader = data.DataLoader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
            )

        evaluator = Evaluator(2)

        model_list = {'DPNet': DPNet(),'FLDPNet': DPNet(), 'unet': UNet(n_channels=3, n_classes=2), 'unetpp': UnetPlusPlus(num_classes=2),'segnet': SegNet(in_channels=3,num_classes=2)}
        model = model_list[args.model_name]

        checkpoints = self.load_model()

        model.load_state_dict(checkpoints["state_dict"])

        model.to(device)
        return device, test_dataloader, evaluator, model, target_paths

    def show_score(self,checkpoint):
        checkpoints = checkpoint
        print("Model Name: %s \n Train Dataset Name: %s \n Test Dataset Name: %s \n " % (self.model_name,self.train_dataset_name,self.test_dataset_name))
        print('Epoch:%d, MIOU:%.2f%% , Dice:%.2f%%' % (checkpoints['Epoch'], checkpoints['MIoU'] * 100,checkpoints['Dice']*100))

    @torch.no_grad()
    def test(self):
        path = os.path.join(os.path.dirname(__file__),r'train_para/ablation_output/%s/Trained on %s/Test on %s' % (self.model_name,self.train_dataset_name,self.test_dataset_name))
        os.makedirs(path,exist_ok=True)
        val_loss = 0.0
        Accuracy = 0.0
        MIoU = 0.0
        Dice = 0.0
        self.model.eval()
        self.evaluator.reset()
        loop = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))
        with torch.no_grad():
            for idx, (img, mask) in loop:
                img, mask = img.to(self.device), mask.to(self.device)
                if self.model_name in ['DPNet',"FLDPNet"]:
                    output = self.model(img, train=False)
                else:
                    output = self.model(img)

                mask_ = torch.argmax(mask, dim=1)
                pred = np.argmax(output.data.cpu().numpy(), axis=1).astype('uint8')
                gt = np.argmax(mask.cpu().numpy(), axis=1)

                loss = structure_loss(output, mask)
                val_loss += loss.item()

                self.evaluator.add_batch(gt, pred)
                acc = self.evaluator.Pixel_Accuracy()
                miou = self.evaluator.BinaryMIOU()
                dice = dice_coeff(output, mask)
                Accuracy += acc
                MIoU += miou
                Dice += dice

                cv2.imwrite(r'./train_para/ablation_output/{}/Trained on {}/Test on {}/{}'.format(self.model_name,self.train_dataset_name,self.test_dataset_name,os.path.basename(self.target_paths[idx])),pred[0]*255)
            test_Accuracy_mean = (Accuracy / len(self.test_dataloader))
            test_MIoU_mean = (MIoU / len(self.test_dataloader))
            test_Dice_mean = (Dice / len(self.test_dataloader))
            print('ACC: %f , MIOU: %f ,Dice: %f' % (test_Accuracy_mean,test_MIoU_mean,test_Dice_mean))

def diceCoeffv2(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred = nn.functional.sigmoid(pred)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return score.sum() / N


def main():
    args = get_args()
    print("Start Time:", end=' ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    predict = Predict(args)
    predict.test()
    print("End Time:", end=' ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

if __name__ == '__main__':
   main()