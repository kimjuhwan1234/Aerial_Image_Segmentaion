from _Transfer_Learning import *
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from segmentation_models_pytorch.losses import DiceLoss

from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import resize
from torchvision.utils import draw_segmentation_masks
from torch.utils.data import Dataset

import os
import cv2
import torch
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

        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

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

        self.additional_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )

    def forward(self, images, masks=None):
        backbone_output = self.backbone(images)

        output = self.additional_layer(backbone_output)

        if masks != None:
            return output, DiceLoss(mode='binary')(output, masks) + nn.BCEWithLogitsLoss()(output, masks)

        return output


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


def get_transform(train):
    transforms = []

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())

    return T.Compose(transforms)


if __name__ == '__main__':
    train_dir = '../AerialImageDatasetrescale/train'
    val_dir = '../AerialImageDatasetrescale/val'
    test_dir = '../AerialImageDatasetrescale/test'

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
                                                           shuffle=False, num_workers=4)
                       }

        model = SegmentationModel()
        model.to(device)
        weight_path = '../weight/Unet.pth'
        print('Finished loading data!')

    train = True
    if train:
        print('Training model...')
        TL = Transfer_Learning(device)
        num_epochs = 10

        optimizer = True
        if optimizer:
            opt = optim.Adam(model.parameters(), lr=0.001)
            # opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        lr = True
        if lr:
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
            print('Check accuracy...')
            correct = 0
            total = 0

            with torch.no_grad():
                for data in dataloaders['val']:
                    images, mask = data[0].to(device), data[1].to(device)
                    output = model(images.to(device))

                    predictions = (output > 0.5).int()
                    correct = (predictions == mask).float()
                    accuracy = correct.mean().item()

            print(f'Accuracy {100 * correct // total} %')

            # plot loss progress
            plt.title("Train-Val Loss")
            plt.plot(range(1, num_epochs + 1), loss_hist["train"], label="train")
            plt.plot(range(1, num_epochs + 1), loss_hist["val"], label="val")
            plt.ylabel("Loss")
            plt.xlabel("Training Epochs")
            plt.legend()
            plt.show()

            # plot accuracy progress
            plt.title("Train-Val Accuracy")
            plt.plot(range(1, num_epochs + 1), metric_hist["train"], label="train")
            plt.plot(range(1, num_epochs + 1), metric_hist["val"], label="val")
            plt.ylabel("Accuracy")
            plt.xlabel("Training Epochs")
            plt.legend()
            plt.show()
            print('Finished accuracy check!')

    if not train:
        print('Skipped train...')
        model.load_state_dict(torch.load(weight_path))
        print('Weights are loaded!')

    Result = False
    if Result:
        print('Try prediction...')
        device = 'cpu'
        model.to(device)

        test_image = read_image("../AerialImageDatasetrescale/check/bellingham1.png")
        test_image = test_image.to(device)
        test_image = resize(test_image, size=(4992, 4992))
        image = test_image
        test_image = test_image.unsqueeze(0)
        eval_transform = get_transform(train=False)

        model.eval()
        with torch.no_grad():
            test_image = eval_transform(test_image)
            predicted_mask = model(test_image)

        original_image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        predicted_mask = (predicted_mask > 0.5).select(1, 0)

        output_image = draw_segmentation_masks(original_image, predicted_mask, alpha=0.5, colors="blue")

        output_image_pil = ToPILImage()(output_image)
        output_image_pil.save("../AerialImageDatasetrescale/output.png")
        print("Masking and saving complete!")

    if not Result:
        print('Show test image...')
        image, mask = test_dataset[1]
        logits_mask = model(image.to(device).unsqueeze(0))  # (c,h,w) -> (b,c,h,w)
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5) * 1.0

        image_pil = ToPILImage()(image.cpu())
        true_mask_pil = ToPILImage()(mask.cpu())
        predicted_mask_pil = ToPILImage()(pred_mask.squeeze().cpu())

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(image_pil)
        axs[0].set_title('Image')

        axs[1].imshow(true_mask_pil)
        axs[1].set_title('True Mask')

        axs[2].imshow(predicted_mask_pil)
        axs[2].set_title('Predicted Mask')

        plt.show()
        print('Completed!')
