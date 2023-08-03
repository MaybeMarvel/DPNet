import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from .metrics import *
from .utils import Evaluator


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target)

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target)
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target)
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class MIoU_Dice_loss(nn.Module):
    def __init__(self,num_classes=2):
        super(MIoU_Dice_loss,self).__init__()
        self.num_classes = num_classes
        self.evaluator = Evaluator(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def forward(self,pred,gt):
        miou_pred = np.argmax(pred.data.cpu().numpy(), axis=1)
        miou_gt = np.argmax(gt.cpu().numpy(), axis=1)
        self.evaluator.reset()
        self.evaluator.add_batch(miou_gt,miou_pred)
        miou = self.evaluator.Mean_Intersection_over_Union()
        miou_score = 1 - miou
        miou_score = torch.tensor(miou_score).to(self.device)
        dice_score = 1 - diceCoeffv2(pred,gt)
        loss = miou_score + dice_score
        return loss
class Focal_loss(nn.Module):
    def __init__(self,gamma=2,alpha=0.5):
        super(Focal_loss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logit, target):
            n, c, h, w = logit.size()
            criterion = nn.CrossEntropyLoss()

            logpt = -criterion(logit, target)
            pt = torch.exp(logpt)
            if self.alpha is not None:
                logpt *= self.alpha
            loss = -((1 - pt) ** self.gamma) * logpt


            return loss

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


class SoftDiceLoss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(SoftDiceLoss,self).__init__()

    def forward(self,logits,targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num,-1)
        m2 = targets.view(num,-1)
        intersection = (m1 * m2)
        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score



class BinarySoftDiceLoss(_Loss):

    def __init__(self):
        super(BinarySoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # print(y_pred.shape,y_true.shape)
        mean_dice = diceCoeffv2(y_pred, y_true)
        return 1 - mean_dice

class dice_bce(nn.Module):
    def __init__(self):
        super(dice_bce,self).__init__()
        self.bdice = BinarySoftDiceLoss()
        self.BCE = nn.BCELoss()
    def forward(self, y_pred, y_true):
        loss = self.BCE(nn.functional.sigmoid(y_pred),y_true) + self.bdice(y_pred,y_true)
        return loss
# class SoftDiceLoss(_Loss):
#
#     def __init__(self, num_classes):
#         super(SoftDiceLoss, self).__init__()
#         self.num_classes = num_classes
#
#     def forward(self, y_pred, y_true):
#         class_dice = []
#         # 从1开始排除背景，前提是颜色表palette中背景放在第一个位置 [[0], ..., ...]
#         for i in range(1, self.num_classes):
#             class_dice.append(diceCoeffv2(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))
#         mean_dice = sum(class_dice) / len(class_dice)
#         return 1 - mean_dice


class SoftDiceLossV2(_Loss):
    def __init__(self, num_classes, weight=[0.73, 0.73, 0.69, 0.93, 0.92], reduction="sum"):
        super(SoftDiceLossV2, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_pred, y_true):
        class_loss = []
        for i in range(1, self.num_classes):
            dice = diceCoeffv2(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :])
            class_loss.append((1-dice) * self.weight[i-1])
        if self.reduction == 'mean':
            return sum(class_loss) / len(class_loss)
        elif self.reduction == 'sum':
            return sum(class_loss)
        else:
            raise NotImplementedError("no such reduction.")


class BinaryTverskyLoss(_Loss):
    def __init__(self, alpha=0.7):
        super(BinaryTverskyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):

        mean_tl = tversky(y_pred, y_true, alpha=self.alpha)
        return 1 - mean_tl


class TverskyLoss(_Loss):
    def __init__(self, num_classes, alpha=0.7):
        super(TverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        tis = []
        for i in range(1, self.num_classes):
            tis.append(tversky(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], alpha=self.alpha))
        ti = sum(tis) / len(tis)
        return 1 - ti


class TverskyLossV2(_Loss):
    def __init__(self, num_classes, alpha=0.7, weight=[0.73, 0.73, 0.69, 0.93, 0.92], reduction="sum"):
        super(TverskyLossV2, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_pred, y_true):
        tls = []
        for i in range(1, self.num_classes):
            dice = tversky(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], alpha=self.alpha)
            tls.append((1 - dice) * self.weight[i-1])
        if self.reduction == 'sum':
            return sum(tls)
        elif self.reduction == 'mean':
            return sum(tls) / len(tls)
        else:
            raise NotImplementedError("no such reduction.")

class BinaryTverskyLoss(_Loss):
    def __init__(self, alpha=0.7):
        super(BinaryTverskyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):

        ti = tversky(y_pred, y_true, alpha=self.alpha)
        return 1 - ti


class FocalTverskyLoss(_Loss):
    def __init__(self, num_classes, gamma=0.75, alpha=0.7):
        super(FocalTverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        tis = []
        for i in range(1, self.num_classes):
            tis.append(tversky(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], alpha=self.alpha))
        ti = sum(tis) / len(tis)
        return torch.pow((1 - ti), self.gamma)


class WBCELoss(_Loss):
    def __init__(self, num_classes,  smooth=0, size=None, weight=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), reduction='mean', ignore_index=255):
        super(WBCELoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weights = None
        if weight:
            weights = []
            w = torch.ones([1, size, size])
            for v in weight:
                weights.append(w * v)
            self.weights = torch.cat(weights, dim=0)
        self.bce_loss = nn.BCELoss(self.weights, reduction, ignore_index)

    def forward(self, inputs, targets):

        return self.bce_loss(inputs, targets * (1 - self.smooth) + self.smooth / self.num_classes)


class BCE_Dice_Loss(_Loss):
    def __init__(self, smooth=0, weight=[1.0, 1.0]):
        super(BCE_Dice_Loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = SoftDiceLoss()
        self.weight = weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        return self.weight[0] * self.bce_loss(nn.functional.sigmoid(inputs), targets * (1 - self.smooth) + self.smooth / 2) + self.weight[1] * self.dice_loss(inputs, targets)


class WBCE_Dice_Loss(_Loss):
    def __init__(self, num_classes, smooth=0, size=None, weight=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
        super(WBCE_Dice_Loss, self).__init__()
        self.wbce_loss = WBCELoss(num_classes=num_classes, smooth=smooth, size=size, weight=weight)
        self.dice_loss = SoftDiceLoss(num_classes=num_classes)

    def forward(self, inputs, targets):
        return self.wbce_loss(inputs, targets) + self.dice_loss(inputs, targets)






if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())