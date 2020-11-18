from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import re
import sys
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

def compute_ROCCurve(gt, pred, class_names):
    n_classes = len(class_names)
    fprs, tprs, thresholds = [], [], []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    color_name =['r','b','k','y','c','g','m','tan','gold','gray','coral','peru','lime','plum']
    for i in range(n_classes):
        fpr, tpr, threshold = roc_curve(gt_np[:, i], pred_np[:, i])
        auc_score = roc_auc_score(gt_np[:, i], pred_np[:, i])#macro
        #select the prediction threshold
        idx = np.where(tpr>auc_score)[0] 
        fpr = fpr[idx]
        tpr = tpr[idx]
        threshold = threshold[idx]
        idx = np.where(fpr<0.2)[0] 
        fprs.append(fpr[idx])
        tprs.append(tpr[idx])
        thresholds.append(threshold[idx])
        
        plt.plot(fpr, tpr, c = color_name[i], ls = '--', label = u'{}{:.4f}'.format(class_names[i],auc_score))
    #plot and save
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right')
    plt.title('ChestXRay8')
    plt.savefig('./Imgs/ROCCurve.jpg')

    return fprs, tpr, thresholds

def compute_AUCs(gt, pred, N_CLASSES=14):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def compute_IoUs_and_Dices(xywh1, xywh2):
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2

    dx = min(x1+w1, x2+w2) - max(x1, x2)
    dy = min(y1+h1, y2+h2) - max(y1, y2)
    intersection = dx * dy if (dx >=0 and dy >= 0) else 0.
    
    union = w1 * h1 + w2 * h2 - intersection
    IoUs = intersection / union
    Dices = 2*IoUs / (IoUs+1)
    return IoUs, Dices

class GradCAM(object):
    """
    1: gradients update when input
    2: backpropatation by the high scores of class
    """
 
    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
 
    def _get_features_hook(self, module, input, output):
        self.feature = output
        #print("feature shape:{}".format(output.size()))
 
    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,length = 1
        :return:
        """
        self.gradient = output_grad[0]
 
    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))
 
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
 
    def __call__(self, inputs, index=None):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()
 
        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]
 
        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
 
        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # nomalization
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 256*256
        cam = cv2.resize(cam, (224, 224)) #resize
        
        return cam
 
 
class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)
 
    def __call__(self, inputs, index=None):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()
 
        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]
 
        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)
 
        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
 
        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU
 
        # nomalization
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize 
        cam = cv2.resize(cam, (224, 224))
        return cam
    
class GuidedBackPropagation(object):
 
    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
                
        self.net.eval()
 
    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """
        :param module:
        :param grad_in: tuple,length=1
        :param grad_out: tuple,length=1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),
 
    def __call__(self, inputs, index=None):
        """
        :param inputs: [1,3,H,W]
        :param index: class_id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
 
        target.backward()
 
        return inputs.grad[0]  # [3,H,W]