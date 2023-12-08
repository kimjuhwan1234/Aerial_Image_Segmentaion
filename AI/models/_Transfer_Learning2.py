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

'''
-  합성곱 신경망의 미세조정(finetuning): 무작위 초기화 대신, 
   신경망을 ImageNet 1000 데이터셋 등으로 미리 학습한 신경망으로 초기화합니다. 
   학습의 나머지 과정들은 평상시와 같습니다.

-  고정된 특징 추출기로써의 합성곱 신경망: 여기서는 마지막에 완전히 연결된 계층을 제외한 
   모든 신경망의 가중치를 고정합니다. 이 마지막의 완전히 연결된 계층은 
   새로운 무작위의 가중치를 갖는 계층으로 대체되어 이 계층만 학습합니다.'''


class Transfer_Learning:
    def __init__(self, device, dataloaders):
        self.device = device
        self.dataloaders = dataloaders

        return

    def imshow(self, inp, title=None):
        """tensor를 입력받아 일반적인 이미지로 보여줍니다."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.

    def get_lr(self, opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def metric_batch(self, output, target):
        # function to calculate metric per mini-batch
        pred = output.argmax(1, keepdim=True)
        corrects = pred.eq(target.view_as(pred)).sum().item()
        return corrects

    def loss_batch(self, loss_func, output, target, opt=None):
        # function to calculate loss per mini-batch
        loss = loss_func(output, target)
        metric_b = self.metric_batch(output, target)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        return loss.item(), metric_b

    def loss_epoch(self, model, loss_func, dataset_dl, sanity_check=False, opt=None):
        running_loss = 0.0
        running_metric = 0.0
        len_data = len(dataset_dl.dataset)

        for xb, yb in tqdm(iter(dataset_dl)):

            xb = xb.to(self.device)
            yb = yb.to(self.device)

            output = model(xb)

            loss_b, metric_b = self.loss_batch(loss_func, output, yb, opt)

            running_loss += loss_b

            if metric_b is not None:
                running_metric += metric_b

            if sanity_check is True:
                break

        loss = running_loss / len_data
        metric = running_metric / len_data

        return loss, metric

    def train_val(self, model, params):
        num_epochs = params['num_epochs']
        loss_func = params["loss_func"]
        opt = params["optimizer"]
        train_dl = params["train_dl"]
        val_dl = params["val_dl"]
        sanity_check = params["sanity_check"]
        lr_scheduler = params["lr_scheduler"]
        path2weights = params["path2weights"]

        loss_history = {'train': [], 'val': []}
        metric_history = {'train': [], 'val': []}

        # # GPU out of memoty error
        # best_model_wts = copy.deepcopy(model.state_dict())

        best_loss = float('inf')

        start_time = time.time()

        for epoch in range(num_epochs):
            current_lr = self.get_lr(opt)
            print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))

            model.train()
            train_loss, train_metric = self.loss_epoch(model, loss_func, train_dl, sanity_check, opt)
            loss_history['train'].append(train_loss)
            metric_history['train'].append(train_metric)

            model.eval()
            with torch.no_grad():
                val_loss, val_metric = self.loss_epoch(model, loss_func, val_dl, sanity_check)
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save(model.state_dict(), path2weights)
                print('Copied best model weights!')
                print('Get best val_loss')

            lr_scheduler.step(val_loss)

            print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' % (
                train_loss, val_loss, 100 * val_metric, (time.time() - start_time) / 60))
            print('-' * 10)

        model.load_state_dict(best_model_wts)

        return model, loss_history, metric_history

        # 모델 예측값 시각화하기


    @torch.no_grad()
    def validate_epoch(self, model: nn.Module, device: torch.device):
        ''' data_loader provides inputs and GTs.
            model receives input train from data_loader and produces logits.
            returns the accuracy of model in percent.
        '''
        model.eval()
        accuracies = []
        for images, labels in tqdm(self.dataloaders, total=len(self.dataloaders), mininterval=1,
                                   desc='measuring accuracy'):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            pred = torch.argmax(logits, dim=1)

            accuracies.append(pred == labels)

        accuracy = torch.concat(accuracies).float().mean() * 100
        return accuracy.item()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10, init_weights=True):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    plt.ion()
    cudnn.benchmark = True
    filterwarnings('ignore')
    multiprocessing.freeze_support()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    Data_loader = True
    if Data_loader:
        data_dir = '../../Reference/hymenoptera_data'
        # data_dir = '/workspace/Reference/hymenoptera_data'
        train_folder = '/train'
        val_folder = '/val'

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}

        RGB = True
        if RGB:
            train_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in image_datasets['train']]
            train_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in image_datasets['train']]

            train_meanR = np.mean([m[0] for m in train_meanRGB])
            train_meanG = np.mean([m[1] for m in train_meanRGB])
            train_meanB = np.mean([m[2] for m in train_meanRGB])
            train_stdR = np.mean([s[0] for s in train_stdRGB])
            train_stdG = np.mean([s[1] for s in train_stdRGB])
            train_stdB = np.mean([s[2] for s in train_stdRGB])

            val_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in image_datasets['val']]
            val_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in image_datasets['val']]

            val_meanR = np.mean([m[0] for m in val_meanRGB])
            val_meanG = np.mean([m[1] for m in val_meanRGB])
            val_meanB = np.mean([m[2] for m in val_meanRGB])
            val_stdR = np.mean([s[0] for s in val_stdRGB])
            val_stdG = np.mean([s[1] for s in val_stdRGB])
            val_stdB = np.mean([s[2] for s in val_stdRGB])

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([val_meanR, val_meanG, val_meanB], [val_stdR, val_stdG, val_stdB])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}

        # create DataLoader
        dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4,
                                                            shuffle=True, num_workers=4),
                       'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4,
                                                          shuffle=False, num_workers=4)}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        class_names = image_datasets['train'].classes

        # 학습 데이터의 배치를 얻습니다.
        inputs, classes = next(iter(dataloaders['train']))

        # 배치로부터 격자 형태의 이미지를 만듭니다.
        out = torchvision.utils.make_grid(inputs)
        TL = Transfer_Learning(device, dataloaders)
        TL.imshow(out, title=[class_names[x] for x in classes])

        num_epochs = 5
        loss_history = {'train': [], 'val': []}
        metric_history = {'train': [], 'val': []}

    finetuning_orignal = False
    if finetuning_orignal:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        # 합성곱 신경망 미세조정(finetuning)
        model_ft = models.resnet18(weights='IMAGENET1K_V1')

        # 여기서 각 출력 샘플의 크기는 2로 설정합니다.
        # 또는, `nn.Linear(num_ftrs, len (class_names))` 로 일반화할 수 있습니다.
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()

        # 모든 매개변수들이 최적화되었는지 관찰
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # 7 에폭마다 0.1씩 학습률 감소
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

        # 학습 및 평가하기
        model_ft, loss_hist, metric_hist = TL.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                                          loss_history, metric_history, num_epochs=num_epochs)

        TL.visualize_model(model_ft)

    finetuning = True
    if finetuning:
        model = resnet18().to(device)
        x = torch.randn(3, 3, 224, 224).to(device)
        output = model(x)
        print(output.size())

        summary(model, (3, 244, 244), device=device.type)
        summary(model, (3, 224, 224), device=device.type)

        loss_func = nn.CrossEntropyLoss(reduction='sum')
        opt = optim.Adam(model.parameters(), lr=0.001)

        custom_lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

        # definc the training parameters
        params_train = {
            'num_epochs': 5,
            'optimizer': opt,
            'loss_func': LabelSmoothingLoss(classes=2, smoothing=0.1),
            'train_dl': dataloaders['train'],
            'val_dl': dataloaders['val'],
            'sanity_check': False,
            'lr_scheduler': custom_lr_scheduler,
            'path2weights': '../resnet18.pth',
        }

        # 'path2weights': '/workspace/data/weight/resnet18.pth',

        model, loss_hist, metric_hist = TL.train_val(model, params_train)

        # TL.visualize_model(model, dataloaders['val'])
        # plt.ioff()
        # plt.show()

    conv_orignal = False
    if conv_orignal:
        # 고정된 특징 추출기로써의 합성곱 신경망
        # 이제, 마지막 계층을 제외한 신경망의 모든 부분을 고정해야 합니다.
        # ``requires_grad = False`` 로 설정하여 매개변수를 고정하여 ``backward()`` 중에
        # 경사도가 계산되지 않도록 해야합니다.

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        # Train-Validation Progress

        model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        for param in model_conv.parameters():
            param.requires_grad = False

        # 새로 생성된 모듈의 매개변수는 기본값이 requires_grad=True 임
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 2)

        model_conv = model_conv.to(device)
        criterion = nn.CrossEntropyLoss()

        # 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

        # 7 에폭마다 0.1씩 학습률 감소
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

        # 학습 및 평가하기
        model_conv, loss_hist, metric_hist = TL.train_model(model_conv, criterion, optimizer_conv,
                                                            exp_lr_scheduler, loss_history, metric_history,
                                                            num_epochs=num_epochs)

        for key in metric_hist:
            metric_hist[key] = metric_hist[key].cpu().numpy()

        for key in loss_hist:
            loss_hist[key] = loss_hist[key].cpu().numpy()

        TL.visualize_model(model_conv)

        plt.ioff()
        plt.show()

    conv = True
    if conv:
        # 고정된 특징 추출기로써의 합성곱 신경망
        # 이제, 마지막 계층을 제외한 신경망의 모든 부분을 고정해야 합니다.
        # requires_grad = False 로 설정하여 매개변수를 고정하여 backward() 중에
        # 경사도가 계산되지 않도록 해야합니다.

        weight_path = 'workspace/data/weight/resnet18.pth'
        model.load_state_dict(torch.load(weight_path))
        model_conv = model
        for param in model_conv.parameters():
            param.requires_grad = False

        # 새로 생성된 모듈의 매개변수는 기본값이 requires_grad=True 임
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 2)

        model_conv = model_conv.to(device)
        criterion = nn.CrossEntropyLoss()

        # 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

        # step_size 에폭마다 0.1씩 학습률 감소
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

        # 학습 및 평가하기
        model_conv = TL.train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, loss_history, metric_history,
                                    num_epochs=5)

        TL.visualize_model(model_conv)
        plt.ioff()
        plt.show()

    accuraccy_check = True
    if accuraccy_check:
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs

        with torch.no_grad():
            for data in dataloaders['val']:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running train through the network
                # outputs = model(train.to(device))
                outputs = model_conv(images.to(device))

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy {100 * correct // total} %')

        # PATH = '../weight/test3.pth'
        # torch.save(model.state_dict(), PATH)

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

    evaluation = True
    if evaluation:

        # 각 클래스별 예측 개수를 세기 위한 변수 초기화
        classes = image_datasets['train'].classes
        correct_pred = {classname: 0 for classname in class_names}
        total_pred = {classname: 0 for classname in class_names}

        # 모델 평가 시에는 그라디언트 계산이 필요 없으므로 torch.no_grad()를 사용합니다
        with torch.no_grad():
            for data in dataloaders['val']:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model_conv(images.to(device))
                _, predictions = torch.max(outputs.data, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[class_names[label]] += 1
                    total_pred[classes[label]] += 1

        # 각 클래스별 정확도 출력
        print("각 클래스별 정확도:")
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'클래스 {classname:5s}의 정확도: {accuracy:.1f}%')

        # 65% 이하의 정확도를 가진 클래스 확인
        low_accuracy_classes = []
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            if accuracy <= 65:
                low_accuracy_classes.append(classname)

        print("65% 이하의 정확도를 가진 클래스:")
        for classname in low_accuracy_classes:
            print(classname)

        incorrect_images = []
        correct_images = []

        with torch.no_grad():
            for data in dataloaders['val']:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model_conv(images.to(device))
                _, predicted = torch.max(outputs.data, 1)

                for i in range(len(predicted)):
                    if predicted[i] != labels[i]:
                        incorrect_images.append(images[i].cpu())
                    else:
                        correct_images.append(images[i].cpu())

        num_classes = 2  # 클래스의 개수에 맞게 설정해주세요

        # 각 클래스에 대해 2개씩 맞은 사진과 틀린 사진 출력
        for label in range(num_classes):
            incorrect_count = 0
            correct_count = 0

            print(f"라벨 {label}에 대한 이미지:")

            # 틀린 사진 출력
            print("틀린 사진들:")
            for img in incorrect_images:
                predicted_label = model_conv(img.unsqueeze(0).to(device)).argmax(dim=1)
                if predicted_label.item() == label and incorrect_count < 2:
                    print("예측한 라벨:", predicted_label.item())
                    plt.imshow(img.permute(1, 2, 0).cpu())
                    plt.pause(0.001)
                    incorrect_count += 1

            # 맞은 사진 출력
            print("맞은 사진들:")
            for img in correct_images:
                predicted_label = model_conv(img.unsqueeze(0).to(device)).argmax(dim=1)
                if predicted_label.item() == label and correct_count < 2:
                    plt.imshow(img.permute(1, 2, 0).cpu())
                    plt.pause(0.001)
                    correct_count += 1

        plt.show()
