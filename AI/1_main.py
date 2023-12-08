from torchvision import tv_tensors
from torchvision.transforms import *
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.transforms import ToPILImage
from torchvision.ops.boxes import masks_to_boxes
from torchvision.models.detection import FasterRCNN
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from torch.backends import cudnn
from warnings import filterwarnings
from torch.utils.data import Dataset
from engine import train_one_epoch, evaluate
from albumentations import OneOf, MotionBlur, OpticalDistortion, GaussNoise, RandomContrast

import os
import torch
import utils
import numpy as np
import torchvision
import matplotlib.pyplot as plt


<<<<<<< HEAD
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
            activation='softmax'
        )

        self.additional_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

    def forward(self, images, masks=None):
        backbone_output = self.backbone(images)

        output = self.additional_layer(backbone_output)

        if masks != None:
            return output, DiceLoss(mode='binary')(output, masks) + nn.BCEWithLogitsLoss()(output, masks)

        return output
=======
class CustomDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
>>>>>>> parent of 0f7aae6 (d)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

<<<<<<< HEAD
def get_train_augs():
    return A.Compose([
        A.Resize(1600, 1600),
        A.Blur(blur_limit=(3, 7)),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50),
        A.VerticalFlip(p=0.5),
        A.GridDistortion(p=0.5, distort_limit=(-0.3, 0.3)),
        A.Rotate(limit=30, p=0.5),  # 최대 30도까지 회전
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    ])
=======
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
>>>>>>> parent of 0f7aae6 (d)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

<<<<<<< HEAD
def get_test_augs():
    return A.Compose([
        A.Resize(1600, 1600),
    ])
=======
        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
>>>>>>> parent of 0f7aae6 (d)


def get_transform(train):
    transforms = []

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())

    return T.Compose(transforms)


def get_transform2(train):
    transforms = []

    if train:
        transforms.append(RandomHorizontalFlip(p=0.5))

        transforms.append(OneOf([
            MotionBlur(),
            OpticalDistortion(),
            GaussNoise(p=0.5),
            RandomContrast()
        ]))

        transforms.append(ElasticTransform())

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(ToTensor())

    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes=2):
    vanilla = True
    if vanilla:
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )

    barrier = False
    if barrier:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)

    return model


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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cpu'
    print(device)

    plt.ion()
    cudnn.benchmark = True
    filterwarnings('ignore')

<<<<<<< HEAD
        train_dataset = CustomDataset(train_dir, get_train_augs())
        val_dataset = CustomDataset(val_dir, get_train_augs())
        test_dataset = CustomDataset(test_dir, get_test_augs())
=======
    Dataset = True
    if Dataset:
        dataset = CustomDataset('../AerialImageDatasetrescale',
                                transforms=get_transform(train=True))
        dataset_test = CustomDataset('../AerialImageDatasetrescale',
                                     get_transform(train=False))
>>>>>>> parent of 0f7aae6 (d)

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        RGB = False
        if RGB:
            print('RGB')
            # train_meanRGB = [np.mean(dataset[i][0].numpy(), axis=(1, 2)) for i in range(len(dataset))]
            # train_stdRGB = [np.std(dataset[i][0].numpy(), axis=(1, 2)) for i in range(len(dataset))]

            for i in range(len(dataset)):
                image, target = dataset[i]

                # Calculate mean and standard deviation for each channel for the current image
                mean_rgb = np.mean(image.numpy(), axis=(1, 2))
                std_rgb = np.std(image.numpy(), axis=(1, 2))

                # Normalize the image based on its own mean and standard deviation
                normalize_transform = transforms.Normalize(mean=mean_rgb, std=std_rgb)
                normalized_image = normalize_transform(image)

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            collate_fn=utils.collate_fn
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=utils.collate_fn
        )

        print('Finished Data loader!')

    # get the model using our helper function
    model = get_model_instance_segmentation()
    model.to(device)

    train = False
    if train:
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        # let's train it for 5 epochs
        num_epochs = 1

        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, 2)
        # model_ft = model_ft.to(device)

        # criterion = nn.CrossEntropyLoss()
        # loss_func = nn.CrossEntropyLoss(reduction='sum')

        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
        # custom_lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            lr_scheduler.step()
            # evaluate(model, data_loader_test, device=device)

        print('Finished train!')

    print("That's it!")

    Save = True
    if Save:
        # 모델 가중치를 저장
        torch.save(model.state_dict(), '../saved_weights.pth')

        # 모델 가중치를 불러오기
        model = get_model_instance_segmentation()
        model.load_state_dict(torch.load('../saved_weights.pth'))
        model.to(device)

        print('Saved!')

    Result = True
    if Result:
        image = read_image("../AerialImageDatasetrescale/test/sfo29.png")
        eval_transform = get_transform(train=False)

        model.eval()
        with torch.no_grad():
<<<<<<< HEAD
            test_image = eval_transform(test_image)
            predicted_mask = model(test_image)
=======
            x = eval_transform(image)
            x = x[:3, ...].to(device)
            predictions = model([x, ])
            pred = predictions[0]
            t = pred['masks']
>>>>>>> parent of 0f7aae6 (d)

        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        image = image[:3, ...]
        pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
        pred_boxes = pred["boxes"].long()
        output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

        # masks = (pred["mask"] > 0.7).squeeze(1)
        masks = (t > 0.7).squeeze(1)
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

        plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0))

        # Convert PyTorch tensor to PIL Image
        output_image_pil = ToPILImage()(output_image)
        # Save the image
        output_image_pil.save("../AerialImageDatasetrescale/output.png")

    evaluate = False
    if evaluate:
        IOU = calculate_iou()
        MAP = calculate_precision_recall()
        Confusion = calculate_confusion_matrix()
        ROC = calculate_roc_curve()
