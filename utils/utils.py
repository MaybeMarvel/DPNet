import numpy as np

#评估器
class Evaluator(object):
    def __init__(self, num_class=32):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)#占位混淆矩阵

    #PA
    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    #MPA
    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    #MIOU
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def BinaryMIOU(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.sum(MIoU)/self.num_class
        return MIoU
    #FWIOU
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


    def _generate_matrix(self, gt_image, pre_image):

        mask = (gt_image >= 0) & (gt_image < self.num_class)

        label = self.num_class * gt_image[mask] + pre_image[mask]

        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix


    def add_batch(self, gt_image, pre_image):

        assert gt_image.shape == pre_image.shape
        for lt, lp in zip(gt_image, pre_image):
            self.confusion_matrix += self._generate_matrix(lt.flatten(), lp.flatten())


    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



if __name__ == '__main__':
    pred = np.array([0, 0, 1, 1, 3, 2])
    groundT = np.array([0, 0, 1, 1, 2, 2])
    eva = Evaluator(4)
    eva.add_batch(groundT,pred)
    pa = eva.Pixel_Accuracy()
    cpa = eva.Pixel_Accuracy_Class()
    miou = eva.Mean_Intersection_over_Union()
    print('pa: %f' % pa)
    print('cpa: %f' % cpa)
    print('miou: %f' % miou)

