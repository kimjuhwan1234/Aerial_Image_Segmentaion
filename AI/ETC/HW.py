from __future__ import print_function, division
from PIL import Image
from tqdm.notebook import tqdm
from torchsummary import summary
from warnings import filterwarnings
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import copy
import time
import torch
import numpy as np
import torchvision
import multiprocessing
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn


# IOU(Intersection over Union)
def calculate_iou(prediction, target):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


# MAP(Mean Average Precision)
def calculate_precision_recall(prediction, target):
    # IOU 계산
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou = np.sum(intersection) / np.sum(union)

    # Precision, Recall 계산
    precision = np.sum(intersection) / np.sum(prediction)
    recall = np.sum(intersection) / np.sum(target)
    return precision, recall


# Confusion Matrix
def calculate_confusion_matrix(prediction, target):
    TP = np.sum(np.logical_and(target == 1, prediction == 1))
    FP = np.sum(np.logical_and(target == 0, prediction == 1))
    FN = np.sum(np.logical_and(target == 1, prediction == 0))
    TN = np.sum(np.logical_and(target == 0, prediction == 0))
    confusion_matrix = np.array([[TP, FP], [FN, TN]])
    return confusion_matrix


# ROC Curve
def calculate_roc_curve(prediction, target):
    # IOU 임계값을 변경하면서 TPR과 FPR 계산
    thresholds = np.linspace(0, 1, 100)  # IOU 임계값 범위 지정
    TPRs, FPRs = [], []
    for threshold in thresholds:
        TP = np.sum(np.logical_and(target == 1, prediction >= threshold))
        FN = np.sum(np.logical_and(target == 1, prediction < threshold))
        FP = np.sum(np.logical_and(target == 0, prediction >= threshold))
        TN = np.sum(np.logical_and(target == 0, prediction < threshold))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPRs.append(TPR)
        FPRs.append(FPR)
    return TPRs, FPRs

if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cpu'
    print(device)

    predictions = torch.randn(1, 3, 224, 224).to(device)
    target = torch.randn(1, 2, 224, 224).to(device)

    TPRs, FPRs = calculate_roc_curve(predictions, target)
    print(TPRs, FPRs)