from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.transforms import ToPILImage
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import resize

from torch.utils.data import Dataset
import segmentation_models_pytorch as smp

from AI.models._Transfer_Learning2 import *


def get_model_instance_segmentation(num_classes=2):
    vanilla = False
    if vanilla:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )

    barrier = True
    if barrier:
        # model=UNet(3,2,True)
        # model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
        #
        # num_ftrs = model.fc.in_features
        #
        # num_ftrs = model.classifier[-1].in_features
        # model.fc = nn.Linear(num_ftrs, 2)
        encoder_architecture = 'resnet18'
        model = smp.Unet(encoder_architecture, classes=num_classes, activation='softmax')
        # model.encoder.load_state_dict(smp.encoders.get_encoder(encoder_architecture, pretrained='imagenet').state_dict())

    return model


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.Resize((4992, 4992)))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


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


class DynamicNormalize(object):
    def __call__(self, img):
        img_np = np.array(img)
        # Calculate mean and std for the current image
        mean = np.mean(img.numpy(), axis=(1, 2))
        std = np.std(img.numpy(), axis=(1, 2))

        # Normalize the image
        normalized_img = transforms.functional.normalize(img, mean, std)

        return normalized_img


if __name__ == '__main__':
    data_dir = '../../AerialImageDatasetrescale/'
    train_folder = '/train'
    val_folder = '/val'

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cpu'
    print(device)

    ETC = True
    if ETC:
        plt.ion()
        cudnn.benchmark = True
        multiprocessing.freeze_support()
        filterwarnings("ignore", category=UserWarning)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    dataload = True
    if dataload:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((4992, 4992)),
                transforms.RandomHorizontalFlip(p=0.5),

                # DynamicNormalize(),
                # T.ToDtype(torch.float, scale=True),
                # T.ToPureTensor(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((4992, 4992)),
                transforms.RandomHorizontalFlip(p=0.5),

                # DynamicNormalize(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}

        dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=2,
                                                            shuffle=True, num_workers=4),
                       'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=2,
                                                          shuffle=False, num_workers=4)}
        # , collate_fn = utils.collate_fn
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        class_names = image_datasets['train'].classes

        inputs, classes = next(iter(dataloaders['train']))

        model = get_model_instance_segmentation()
        model.to(device)

    train = True
    if train:
        TL = Transfer_Learning(device, dataloaders)
        # out = torchvision.utils.make_grid(inputs)
        # TL.imshow(out, title=[class_names[x] for x in classes])
        weight_path='../resnet18.pth'
        num_epochs = 1

        # x = torch.randn(2, 3, 4992, 4992).to(device)
        # output = model(x)
        # print(output.size())
        #
        # summary(model, (3, 244, 244), device=device.type)

        # construct an optimizer
        # params = [p for p in model.parameters() if p.requires_grad]
        #
        # opt = torch.optim.SGD(
        #     params,
        #     lr=0.005,
        #     momentum=0.9,
        #     weight_decay=0.0005
        # )
        # opt = optim.Adam(model.parameters(), lr=0.001)
        opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=3,
            gamma=0.1
        )
        # lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

        criterion = nn.CrossEntropyLoss(reduction='sum')
        # criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)



        params_train = {
            'path2weights': weight_path,
            'num_epochs': num_epochs,
            'optimizer': opt,
            'loss_func': criterion,
            'lr_scheduler': lr_scheduler,

            'train_dl': dataloaders['train'],
            'val_dl': dataloaders['val'],
            'sanity_check': False,

        }

        model, loss_hist, metric_hist = TL.train_val(model, params_train)

    Save = False
    if Save:
        # 모델 가중치를 저장
        torch.save(model.state_dict(), 'saved_weights.pth')

        # 모델 가중치를 불러오기
        model = get_model_instance_segmentation()
        model.load_state_dict(torch.load('saved_weights.pth'))
        model.to(device)

    Result = True
    if Result:
        torch.cuda.empty_cache()
        test_image = read_image("../../AerialImageDatasetrescale/check/bellingham1.png")
        test_image = test_image.to(device)
        eval_transform = get_transform(train=False)
        image = test_image
        image = resize(image, size=(4992, 4992))
        test_image = test_image.unsqueeze(0)

        model.eval()

        with torch.no_grad():
            test_image = eval_transform(test_image)
            predicted_mask = model(test_image)

        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        predicted_mask = (predicted_mask > 0.5).select(1, 0)
        output_image = draw_segmentation_masks(image, predicted_mask, alpha=0.5, colors="blue")
        # Save the predicted mask

        # Convert PyTorch tensor to PIL Image
        output_image_pil = ToPILImage()(output_image)
        # Save the image
        output_image_pil.save("../AerialImageDatasetrescale/output2.png")

        print("Masking and saving complete.")

# RGB = True
# if RGB:
#     train_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in image_datasets['train']]
#     train_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in image_datasets['train']]
#
#     # for i in range(len(dataset)):
#     #     image, target = dataset[i]
#     #
#     #     # Calculate mean and standard deviation for each channel for the current image
#     #     mean_rgb = np.mean(image.numpy(), axis=(1, 2))
#     #     std_rgb = np.std(image.numpy(), axis=(1, 2))
#     #
#     #     # Normalize the image based on its own mean and standard deviation
#     #     normalize_transform = transforms.Normalize(mean=mean_rgb, std=std_rgb)
#     #     normalized_image = normalize_transform(image)
#
#     train_meanR = np.mean([m[0] for m in train_meanRGB])
#     train_meanG = np.mean([m[1] for m in train_meanRGB])
#     train_meanB = np.mean([m[2] for m in train_meanRGB])
#     train_stdR = np.mean([s[0] for s in train_stdRGB])
#     train_stdG = np.mean([s[1] for s in train_stdRGB])
#     train_stdB = np.mean([s[2] for s in train_stdRGB])
#
#     val_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in image_datasets['val']]
#     val_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in image_datasets['val']]
#
#     val_meanR = np.mean([m[0] for m in val_meanRGB])
#     val_meanG = np.mean([m[1] for m in val_meanRGB])
#     val_meanB = np.mean([m[2] for m in val_meanRGB])
#     val_stdR = np.mean([s[0] for s in val_stdRGB])
#     val_stdG = np.mean([s[1] for s in val_stdRGB])
#     val_stdB = np.mean([s[2] for s in val_stdRGB])
#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([val_meanR, val_meanG, val_meanB], [val_stdR, val_stdG, val_stdB])
#     ]),
# }
#
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                   for x in ['train', 'val']}


    evaluate = False
    if evaluate:
        IOU = calculate_iou()
        MAP = calculate_precision_recall()
        Confusion = calculate_confusion_matrix()
        ROC = calculate_roc_curve()