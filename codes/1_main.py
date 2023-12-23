from _Transfer_Learning import *
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from segmentation_models_pytorch.losses import DiceLoss

from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image, resize
from torchvision.utils import draw_segmentation_masks

import os
import cv2
import torch
import warnings
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.optim as optim
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


class CustomDataset(Dataset):
    def __init__(self, path, augmentations=None):
        self.image_path = os.path.join(path, 'image')
        self.mask_path = os.path.join(path, 'mask')
        self.augmentations = augmentations
        self.file_list = os.listdir(self.image_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_filename = os.path.join(self.image_path, self.file_list[index])
        mask_filename = os.path.join(self.mask_path, self.file_list[index])

        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)  # (h,w)
        mask = np.expand_dims(mask, axis=-1)  # (h,w,c)

        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        encoder = 'resnet50'
        weights = 'imagenet'
        self.backbone = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=1,
            activation=None
        )

        # self.backbone = smp.DeepLabV3Plus(
        #     encoder_name=encoder,
        #     encoder_weights=weights,
        #     in_channels=3,
        #     classes=1,
        #     activation=None
        # )

        self.additional_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )

    def forward(self, images, masks=None):
        backbone_output = self.backbone(images)

        output = self.additional_layer(backbone_output)

        if masks != None:
            return output, DiceLoss(mode='binary')(output, masks) + nn.BCEWithLogitsLoss()(output, masks)
            # return backbone_output, DiceLoss(mode='binary')(backbone_output, masks) + nn.BCEWithLogitsLoss()(backbone_output, masks)

        return output


def calculate_iou(gt, prediction):
    # IOU(Intersection over Union)
    intersection = np.logical_and(gt, prediction)
    union = np.logical_or(gt, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score  # IOU(Intersection over Union)


def calculate_precision_recall(gt, prediction):
    # MAP(Mean Average Precision)
    intersection = np.logical_and(gt, prediction)
    union = np.logical_or(gt, prediction)

    # Precision, Recall 계산
    precision = np.sum(intersection) / np.sum(prediction)
    recall = np.sum(intersection) / np.sum(gt)
    return precision, recall


def calculate_confusion_matrix(gt, prediction):
    # Confusion Matrix
    TP = np.sum(np.logical_and(gt == 1, prediction == 1))
    FP = np.sum(np.logical_and(gt == 0, prediction == 1))
    FN = np.sum(np.logical_and(gt == 1, prediction == 0))
    TN = np.sum(np.logical_and(gt == 0, prediction == 0))
    confusion_matrix = pd.DataFrame([[TP, FP], [TN, FN]], columns=['T', 'F'], index=['P', 'N'])
    return confusion_matrix


def calculate_roc_curve(gt, prediction):
    # ROC Curve
    thresholds = np.linspace(0, 1, 100)
    TPRs, FPRs = [], []
    for threshold in thresholds:
        TP = np.sum(np.logical_and(gt == 1, prediction >= threshold))
        FN = np.sum(np.logical_and(gt == 1, prediction < threshold))
        FP = np.sum(np.logical_and(gt == 0, prediction >= threshold))
        TN = np.sum(np.logical_and(gt == 0, prediction < threshold))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPRs.append(TPR)
        FPRs.append(FPR)

    # ROC curve 그리기
    plt.plot(FPRs, TPRs, label='ROC curve')
    plt.plot([0, 1], [0, 1], '--', label='Random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def get_train_augs():
    return A.Compose([
        A.CenterCrop(512, 512),
        A.Blur(blur_limit=(3, 7)),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50),
        A.VerticalFlip(p=0.5),
        A.GridDistortion(p=0.5, distort_limit=(-0.3, 0.3)),
        A.Rotate(limit=30, p=0.5),  # 최대 30도까지 회전
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    ])


def get_test_augs():
    return A.Compose([
        A.CenterCrop(512, 512),
    ])


def get_transform():
    transforms = []
    transforms.append(T.CenterCrop((4992, 4992)))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


if __name__ == '__main__':
    train_dir = '../database/train'
    val_dir = '../database/val'
    test_dir = '../database/test'
    check_dir = '../database/check'

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    dataload = True
    if dataload:
        print('Loading data...')

        train_dataset = CustomDataset(train_dir, get_train_augs())
        val_dataset = CustomDataset(val_dir, get_train_augs())
        test_dataset = CustomDataset(test_dir, get_test_augs())

        dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=2,
                                                            shuffle=True, num_workers=4),
                       'val': torch.utils.data.DataLoader(val_dataset, batch_size=2,
                                                          shuffle=False, num_workers=4),
                       'test': torch.utils.data.DataLoader(test_dataset, batch_size=2,
                                                           shuffle=False, num_workers=4),
                       }

        model = SegmentationModel()
        model.to(device)
        weight_path = '../weight/Unet_custom_30.pth'
        print('Finished loading data!')

    train = False
    if train:
        print('Training model...')
        TL = Transfer_Learning(device)
        num_epochs = 10

        # opt = optim.Adam(model.parameters(), lr=0.01)
        opt = optim.Adagrad(model.parameters(),lr=0.01)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

        params_train = {
            'num_epochs': num_epochs,
            'weight_path': weight_path,

            'train_dl': dataloaders['train'],
            'val_dl': dataloaders['val'],

            'optimizer': opt,
            'lr_scheduler': lr_scheduler,
        }

        model, loss_hist, metric_hist = TL.train_val(model, params_train)
        print('Finished training model!')

        accuracy_check = True
        if accuracy_check:
            print('Check stats...')
            correct = 0
            total = 0

            with torch.no_grad():
                for data in dataloaders['val']:
                    images, mask = data[0].to(device), data[1].to(device)
                    output = model(images)

                    pred_mask = (output > 0.5).float()
                    accuracy = (pred_mask == mask).sum().item()
                    total_pixels = mask.numel()
                    accuracy = accuracy / total_pixels

            print(f'Accuracy {100 * accuracy:.2f} %')

            loss_hist_numpy = loss_hist.applymap(
                lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)
            metric_hist_numpy = metric_hist.applymap(
                lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)

            # plot loss progress
            plt.title("Train-Val Loss")
            plt.plot(range(1, num_epochs + 1), loss_hist_numpy.iloc[:, 0], label="train")
            plt.plot(range(1, num_epochs + 1), loss_hist_numpy.iloc[:, 1], label="val")
            plt.ylabel("Loss")
            plt.xlabel("Training Epochs")
            plt.legend()
            plt.show()

            # plot accuracy progress
            plt.title("Train-Val Accuracy")
            plt.plot(range(1, num_epochs + 1), metric_hist_numpy.iloc[:, 0], label="train")
            plt.plot(range(1, num_epochs + 1), metric_hist_numpy.iloc[:, 1], label="val")
            plt.ylabel("Accuracy")
            plt.xlabel("Training Epochs")
            plt.legend()
            plt.show()
            print('Finished stats check!')

    if not train:
        print('Skipped train...')
        model.load_state_dict(torch.load(weight_path))
        print('Weights are loaded!')

    Prediction = False
    if Prediction:
        print('Try prediction...')
        device = 'cpu'
        model.to(device)
        warnings.filterwarnings("ignore")
        test_image = read_image("../database/check/sfo29.png")
        test_image = test_image.to(device)

        image = resize(test_image, (4992, 4992))

        eval_transform = get_transform()
        test_image = eval_transform(test_image).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            predicted_mask = model(test_image)
            predicted_mask = torch.sigmoid(predicted_mask)

        original_image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        predicted_mask = (predicted_mask > 0.8).select(1, 0)

        output_image = draw_segmentation_masks(original_image, predicted_mask, alpha=0.5, colors="blue")

        output_image = to_pil_image(output_image)
        output_image.save("../database/output.png")
        print("Masking and saving complete!")

    if not Prediction:
        print('Evaluation in progress for testset...')
        image, mask = test_dataset[9]
        logits_mask = model(image.to(device).unsqueeze(0))  # (c,h,w) -> (b,c,h,w)
        logits_mask = torch.sigmoid(logits_mask)
        pred_mask = (logits_mask > 0.5) * 1.0
        pred = pred_mask.squeeze(0)

        image_pil = ToPILImage()(image.cpu())
        true_mask_pil = ToPILImage()(mask.cpu())
        predicted_mask_pil = ToPILImage()(pred.cpu())

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(image_pil)
        axs[0].set_title('Image')

        axs[1].imshow(true_mask_pil)
        axs[1].set_title('True Mask')

        axs[2].imshow(predicted_mask_pil)
        axs[2].set_title('Predicted Mask')

        plt.show()

        evaluate = True
        if evaluate:
            device = 'cpu'
            mask = mask.to(device).numpy()  # ground truth
            logits_mask = logits_mask.to(device).detach().numpy() # predicted probabilities
            pred_mask = pred_mask.to(device).numpy()  # predicted mask consisted with 0 or 1

            calculate_roc_curve(mask, logits_mask)
            iou = calculate_iou(mask, pred_mask)
            precision, recall = calculate_precision_recall(mask, pred_mask)
            cfx = calculate_confusion_matrix(mask, pred_mask)

            print(f'IOU score: {iou:.2f}')
            print(f'Precision:{precision:.2f}')
            print(f'Recall:{recall:.2f}')
            print(f'Confusion matrix:')
            print(cfx)
            print('Completed!')
