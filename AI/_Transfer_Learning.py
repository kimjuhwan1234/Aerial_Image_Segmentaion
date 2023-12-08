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
    def __init__(self, device, dataloaders, datasize, classnames):
        self.device = device
        self.val_dl = dataloaders['val']
        self.dataloaders = dataloaders
        self.dataset_sizes = datasize
        self.classnames = classnames
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

            xb = xb.to(device)
            yb = yb.to(device)
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

    def train_model(self, model, criterion, optimizer, scheduler, loss_history, metric_history, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 학습 모드로 설정
                else:
                    model.eval()  # 모델을 평가 모드로 설정

                running_loss = 0.0
                running_corrects = 0

                # 데이터를 반복
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # 매개변수 경사도를 0으로 설정
                    optimizer.zero_grad()

                    # 순전파
                    # 학습 시에만 연산 기록을 추적
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 학습 단계인 경우 역전파 + 최적화
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 통계
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    loss_history['val'].append(epoch_loss)
                    metric_history['val'].append(epoch_acc)

                if phase == 'train':
                    loss_history['train'].append(epoch_loss)
                    metric_history['train'].append(epoch_acc)

                # 모델을 깊은 복사(deep copy)함
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # 가장 나은 모델 가중치를 불러옴
        model.load_state_dict(best_model_wts)

        return model, loss_history, metric_history

        # 모델 예측값 시각화하기

<<<<<<< HEAD
class CustomDataset(Dataset):
    def __init__(self, root_dir, augmentations_image):
        self.image_folder = root_dir + '/image'
        self.mask_folder = root_dir + '/mask'
        self.augmentations_image = augmentations_image
=======
    def visualize_model(self, model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
>>>>>>> parent of 0f7aae6 (d)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.val_dl):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {self.classnames[preds[j]]}')
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

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

<<<<<<< HEAD
        data = self.augmentations_image(image=image, mask=mask)

        image = data['image']
        mask = data['mask']
=======
            logits = model(images)
            pred = torch.argmax(logits, dim=1)

            accuracies.append(pred == labels)
>>>>>>> parent of 0f7aae6 (d)

        accuracy = torch.concat(accuracies).float().mean() * 100
        return accuracy.item()

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
        # data_dir = '../AerialImageDatasetrescale/'
        data_dir = '../Reference/hymenoptera_data'
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

        dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                   for x in ['train', 'val']}

        RGB = False
        if RGB:
            train_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset['train']]
            train_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset['train']]

            train_meanR = np.mean([m[0] for m in train_meanRGB])
            train_meanG = np.mean([m[1] for m in train_meanRGB])
            train_meanB = np.mean([m[2] for m in train_meanRGB])
            train_stdR = np.mean([s[0] for s in train_stdRGB])
            train_stdG = np.mean([s[1] for s in train_stdRGB])
            train_stdB = np.mean([s[2] for s in train_stdRGB])

            val_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset['val']]
            val_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset['val']]

            val_meanR = np.mean([m[0] for m in val_meanRGB])
            val_meanG = np.mean([m[1] for m in val_meanRGB])
            val_meanB = np.mean([m[2] for m in val_meanRGB])
            val_stdR = np.mean([s[0] for s in val_stdRGB])
            val_stdG = np.mean([s[1] for s in val_stdRGB])
            val_stdB = np.mean([s[2] for s in val_stdRGB])

        RGB_mean=[0.39189383, 0.39703637, 0.34817487]
        RGB_std=[0.25606027, 0.24655445, 0.24031337]

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(RGB_mean, RGB_std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(RGB_mean, RGB_std)
            ]),
        }

        dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                   for x in ['train', 'val']}

        # create DataLoader
        dataloaders = {'train': torch.utils.data.DataLoader(dataset['train'], batch_size=4,
                                                            shuffle=True, num_workers=4),
                       'val': torch.utils.data.DataLoader(dataset['val'], batch_size=4,
                                                          shuffle=False, num_workers=4)}

        dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}

        class_names = dataset['train'].classes

        # 학습 데이터의 배치를 얻습니다.
        inputs, classes = next(iter(dataloaders['train']))

        # 배치로부터 격자 형태의 이미지를 만듭니다.
        out = torchvision.utils.make_grid(inputs)
        TL = Transfer_Learning(device, dataloaders, dataset_sizes, class_names)
        TL.imshow(out, title=[class_names[x] for x in classes])

        num_epochs = 5
        loss_history = {'train': [], 'val': []}
        metric_history = {'train': [], 'val': []}

    finetuning_orignal = True
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
        # loss_func = nn.CrossEntropyLoss(reduction='sum')

        # 모든 매개변수들이 최적화되었는지 관찰
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # 7 에폭마다 0.1씩 학습률 감소
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
        # custom_lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

        # 학습 및 평가하기
        model_ft, loss_hist, metric_hist = TL.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                                          loss_history, metric_history, num_epochs=num_epochs)

        torch.save(model_ft.state_dict(), 'model_weights.pth')

        TL.visualize_model(model_ft)

    conv_orignal = True
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
        classes = dataset['train'].classes
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
