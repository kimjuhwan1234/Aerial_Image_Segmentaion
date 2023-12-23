import pandas as pd
from tqdm.notebook import tqdm

import time
import torch
import torch.nn as nn

if True:
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
                    nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
                              bias=False),
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
                    nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride,
                              bias=False),
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

class Transfer_Learning:
    def __init__(self, device):
        self.device = device

    def get_lr(self, opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def metric_batch(self, output, mask):
        pred_mask = (output > 0.5).float()  # 마스크 예측 값 이진화
        accuracy = (pred_mask == mask).sum().item()  # 정확한 픽셀 수 계산
        total_pixels = mask.numel()  # 전체 픽셀 수 계산
        accuracy = accuracy / total_pixels

        return accuracy

    def eval_epoch(self, model, dataset_dl):
        total_loss = 0.0
        total_metric = 0.0
        len_data = len(dataset_dl.dataset)

        model.eval()

        with torch.no_grad():
            for image, mask in tqdm(dataset_dl):
                image = image.to(self.device)
                mask = mask.to(self.device)

                output, loss = model(image, mask)
                output = torch.sigmoid(output)
                total_loss += loss

                metric = self.metric_batch(output, mask)
                total_metric += metric

            total_loss = total_loss / len_data
            total_metric = total_metric / len_data

        return total_loss, total_metric

    def loss_epoch(self, model, dataset_dl, opt):
        total_loss = 0.0
        total_metric = 0.0
        len_data = len(dataset_dl.dataset)

        model.train()

        for image, mask in tqdm(dataset_dl):
            image = image.to(self.device)
            mask = mask.to(self.device)

            opt.zero_grad()
            output, loss = model(image, mask)
            output = torch.sigmoid(output)
            loss.backward()
            opt.step()

            total_loss += loss

            metric = self.metric_batch(output, mask)
            total_metric += metric

        total_loss = total_loss / len_data
        total_metric = total_metric / len_data

        return total_loss, total_metric

    def train_val(self, model, params):
        num_epochs = params['num_epochs']
        weight_path = params["weight_path"]

        train_dl = params["train_dl"]
        val_dl = params["val_dl"]

        opt = params["optimizer"]
        lr_scheduler = params["lr_scheduler"]

        loss_history = pd.DataFrame(columns=['train', 'val'])
        metric_history = pd.DataFrame(columns=['train', 'val'])

        best_loss = float('inf')
        start_time = time.time()

        for epoch in range(num_epochs):
            current_lr = self.get_lr(opt)
            print(f'Epoch {epoch + 1}/{num_epochs}, current lr={current_lr}')

            train_loss, train_metric = self.loss_epoch(model, train_dl, opt)

            loss_history.loc[epoch, 'train'] = train_loss
            metric_history.loc[epoch, 'train'] = train_metric

            val_loss, val_metric = self.eval_epoch(model, val_dl)
            loss_history.loc[epoch, 'val'] = val_loss
            metric_history.loc[epoch, 'val'] = val_metric

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), weight_path)
                print('Saved model weight!')

            lr_scheduler.step(val_loss)

            print(f'train loss: {train_loss:.2f}, val loss: {val_loss:.2f}')
            print(f'accuracy: {100 * val_metric:.2f} %, time: {(time.time() - start_time) / 60:.2f}')

            print('-' * 10)

        return model, loss_history, metric_history
