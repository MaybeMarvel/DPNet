import os
import numpy as np
import torch
import torch.optim as optim
from model.DPNet import DPNet
from model.ablation_model.unet import UNet
from model.ablation_model.unetpp import UnetPlusPlus
from model.ablation_model.segnet import SegNet
from utils.metrics import diceCoeffv2
from utils.utils import Evaluator
from utils.loss import structure_loss
from tqdm import tqdm
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import glob
from data.MedicalDataset import get_dataloaders


def get_args():
    parser = argparse.ArgumentParser(description="Train DPNet on specified dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir", "CVC", "HAM"])
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument("--model_name", type=str, default= 'DPNet', dest='model_name')

    return parser.parse_args()


class TrainerV1(object):
    def __init__(self,args):
        super(TrainerV1,self).__init__()
        self.args = args
        self.dataset_name = self.args.dataset
        self.model_name = args.model_name

        (
            self.device,
            self.train_dataloader,
            self.val_dataloader,
            self.evaluator,
            self.model,
            self.optimizer,
        ) = self.build()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='max',factor=0.5,min_lr=1e-6,patience=5)

        self.dataset = self.args.dataset
        self.epoch = self.args.epochs
        self.best_score = 0.0
        self.count = 0
        self.init_train_para()

    def init_train_para(self):
        print('\n\nTrainning Model: %s \nTrainning Dataset: %s \nEpoch: %d \n\n' % (self.model_name,self.dataset_name,self.epoch))

    def run(self):
        for epoch in range(self.epoch):
            print("-----------------------------\nTrain Data Epoch[%d/%d]:" % (epoch + 1, self.epoch))
            self.train(self.train_dataloader)
            val_Dice_mean = self.val(epoch,self.val_dataloader)
            if self.early_stop(lr=self.optimizer.state_dict()['param_groups'][0]['lr'], score=val_Dice_mean):
                break

    def build(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        if self.args.dataset == "Kvasir" or "HAM":
            img_path = self.args.root + "images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = self.args.root + "masks/*"
            target_paths = sorted(glob.glob(depth_path))
        elif self.args.dataset == "CVC":
            img_path = self.args.root + "Original/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = self.args.root + "Ground Truth/*"
            target_paths = sorted(glob.glob(depth_path))
        train_dataloader, _, val_dataloader = get_dataloaders(
            input_paths, target_paths, batch_size=self.args.batch_size,dataset=self.args.dataset
        )

        model_list = {'DPNet': DPNet(), 'unet':UNet(n_channels=3,n_classes=2), 'unetpp':UnetPlusPlus(num_classes=2),'segnet':SegNet(in_channels=3,num_classes=2)}
        evaluator = Evaluator(2)
        model = model_list[self.args.model_name]
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,betas=(0.9,0.999))

        return (
            device,
            train_dataloader,
            val_dataloader,
            evaluator,
            model,
            optimizer,
        )

    def train(self,train_dataloader):
            #train
            train_time = time.time()
            rate_size = [0.75,1,1.25]
            train_loss = 0.0
            Accuracy = 0.0
            MIoU = 0.0
            Dice = 0.0
            self.model.train()

            loop = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
            for idx,(img,mask) in loop:
                # for rate in rate_size:
                    img,mask = img.to(self.device),mask.to(self.device)
                    # trainsize = int(round(img.size(2) * rate / 32) * 32)
                    # if rate !=1:
                    # img = nn.functional.upsample(img, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    # mask = nn.functional.upsample(mask, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    output = self.model(img)


                    # loss = self.criterion(output, mask_.long())
                    if self.model_name in ['DPNet_withoutAA', 'DPNet']:
                        pred = np.argmax(output[0].data.cpu().numpy(), axis=1)
                        gt = np.argmax(mask.cpu().numpy(), axis=1)
                        loss = structure_loss(output[0],mask) + structure_loss(output[1],mask) + structure_loss(output[2],mask) + structure_loss(output[3],mask)
                        dice = diceCoeffv2(output[0], mask)
                    else:
                        pred = np.argmax(output.data.cpu().numpy(), axis=1)
                        gt = np.argmax(mask.cpu().numpy(), axis=1)
                        loss = structure_loss(output,mask)
                        dice = diceCoeffv2(output, mask)
                    train_loss += loss.item()
                    loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # if rate == 1:
                    self.evaluator.reset()
                    self.evaluator.add_batch(gt, pred)
                    acc = self.evaluator.Pixel_Accuracy()
                    miou = self.evaluator.BinaryMIOU()
                    Accuracy += acc
                    MIoU += miou
                    Dice += dice

            train_loss_mean = (train_loss / len(train_dataloader))
            train_Accuracy_mean = (Accuracy / len(train_dataloader))
            train_MIoU_mean = (MIoU / len(train_dataloader))
            train_Dice_mean = (Dice / len(train_dataloader))

            print("Train    : Loss:%.4f LR:%f Accuracy:%.4f MIoU:%.4f Dice:%.4f Time:%ds"
                    % (train_loss_mean,self.optimizer.state_dict()['param_groups'][0]['lr'],train_Accuracy_mean,train_MIoU_mean,train_Dice_mean,(time.time()-train_time)))

    @torch.no_grad()
    def val(self,epoch,val_loader):
        # val
        val_time = time.time()
        val_loss = 0.0
        Accuracy = 0.0
        MIoU = 0.0
        Dice = 0.0
        self.model.eval()
        self.evaluator.reset()
        with torch.no_grad():
            loop = tqdm(enumerate(val_loader), total=len(val_loader))
            for idx, (img, mask) in loop:
                img, mask = img.to(self.device), mask.to(self.device)
                # with torch.no_grad():

                # loss = self.criterion(output,mask_.long())
                if self.model_name in ['DPNet_withoutAA', 'DPNet']:
                    output = self.model(img, train=False)
                    pred = np.argmax(output.data.cpu().numpy(), axis=1).astype('uint8')
                    gt = np.argmax(mask.cpu().numpy(), axis=1)
                    loss = structure_loss(output, mask)
                else:
                    output = self.model(img)
                    pred = np.argmax(output.data.cpu().numpy(), axis=1).astype('uint8')
                    gt = np.argmax(mask.cpu().numpy(), axis=1)
                    loss = structure_loss(output, mask)
                val_loss += loss.item()

                self.evaluator.add_batch(gt, pred)
                acc = self.evaluator.Pixel_Accuracy()
                miou = self.evaluator.BinaryMIOU()
                dice = diceCoeffv2(output, mask)
                Accuracy += acc
                MIoU += miou
                Dice += dice

            val_loss_mean = (val_loss / len(val_loader))
            val_Accuracy_mean = (Accuracy / len(val_loader))
            val_MIoU_mean = (MIoU / len(val_loader))
            val_Dice_mean = (Dice / len(val_loader))

            print("Valition : Loss:%.4f LR:%f Accuracy:%.4f MIoU:%.4f Dice:%.4f Time:%ds" % (
                val_loss_mean, self.optimizer.state_dict()['param_groups'][0]['lr'], val_Accuracy_mean, val_MIoU_mean,
                val_Dice_mean, (time.time() - val_time)))
            # if (epoch + 1) % 1 == 0:
            if self.model_name in ['DPNet', 'DPNet_withoutAA'] and ((epoch + 1) % 5) == 0:
                save_pic(self.dataset_name, epoch, gt, pred, self.model_name)

            if val_Dice_mean > self.best_score:
                if self.model_name in ['DPNet', 'DPNet_withoutAA']:
                    print('Saving model...', end='\t')
                    self.save_model(epoch, val_loss_mean, val_Accuracy_mean, val_MIoU_mean, val_Dice_mean)
                    print('Save model Succeed')

                else:
                    print('Saving model...', end='\t')
                    os.makedirs(os.path.join(os.path.dirname(__file__), r'train_para/checkpoints/%s' % self.dataset),
                                exist_ok=True)
                    save_path = os.path.join(os.path.dirname(__file__),
                                             r'train_para/checkpoints/%s/%s.pth.tar' % (
                                             self.dataset, self.dataset + '_' + self.model_name))
                    model_dict = {'Epoch': epoch + 1,
                                  'Loss': val_loss_mean,
                                  'Accuracy': val_Accuracy_mean,
                                  'MIoU': val_MIoU_mean,
                                  'Dice': val_Dice_mean,
                                  'state_dict': self.model.state_dict(),
                                  'optimizer': self.optimizer.state_dict(), }
                    torch.save(model_dict, f=save_path)
                    self.best_score = val_Dice_mean
                    print('Save model Succeed')

        self.scheduler.step(val_Dice_mean)
        return val_Dice_mean

    def early_stop(self,lr,score):
        if lr == 1e-6 and self.best_score >= score:
            self.count += 1
            if self.count >= 10:
                return True
            else:
                return False

    def save_model(self,epoch,val_loss_mean,val_Accuracy_mean,val_MIoU_mean,val_Dice_mean):
        os.makedirs(os.path.join(os.path.dirname(__file__), r'train_para/checkpoints/%s' % self.dataset),
                        exist_ok=True)
        save_path = os.path.join(os.path.dirname(__file__),
                                     r'train_para/checkpoints/%s/%s.pth.tar' % (self.dataset, self.dataset + '_' +self.model_name))
        self.best_score = val_Dice_mean
        model_dict = {'Epoch': epoch + 1,
                          'Loss': val_loss_mean,
                          'Accuracy': val_Accuracy_mean,
                          'MIoU': val_MIoU_mean,
                          'Dice': val_Dice_mean,
                          'state_dict': self.model.state_dict(),
                          'optimizer': self.optimizer.state_dict(),}
        torch.save(model_dict,f=save_path)



def show_socre(dataset_name,model_name):
    if dataset_name in ['Kvasir','CVC']:
        os.makedirs(os.path.join(os.path.dirname(__file__),r'train_para/checkpoints/%s' % dataset_name),exist_ok=True)
        path = os.path.join(os.path.dirname(__file__),r'train_para/checkpoints/%s/%s.pth.tar'% (dataset_name,dataset_name + '_' + model_name))
        model = torch.load(f=path)
        print(model['Epoch'],model['Loss'],model['MIoU'],model['Dice'])


def save_pic(dataset_name,epoch,gt,pred,model_name):
    if dataset_name in ['Kvasir','CVC']:
        path = os.path.join(os.path.dirname(__file__),
                            r'train_para/trainpic/%s/%s' % (dataset_name,model_name))
        os.makedirs(path, exist_ok=True)
        plt.subplot(121)
        plt.imshow(gt[0], cmap='gray')
        plt.subplot(122)
        plt.imshow(pred[0], cmap='gray')
        plt.savefig(path + r'/%d' % (epoch + 1))
        plt.close('all')


def train(args):
    print("Start Time:",end=' ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    trainer = TrainerV1(args)
    trainer.run()
    print("End Time:",end=' ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    show_socre(args.dataset,args.model_name)


if __name__ == '__main__':
    args = get_args()
    train(args)
    # for scale in [0.75,1.0,1.5]:
    #     train_size = torch.nn.functional.interpolate()