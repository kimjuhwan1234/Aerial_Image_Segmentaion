import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def train_loop(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # 예측(prediction)과 손실(loss) 계산
            pred = model(X)
            loss = loss_fn(pred, y)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


NN = NeuralNetwork()
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    NN = NeuralNetwork().to(device)
    print(NN)

    X = torch.rand(1, 28, 28, device=device)
    logits = NN(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    for name, param in NN.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    '''학습 시에는 다음과 같은 하이퍼파라미터를 정의합니다:
     - 에폭(epoch) 수 - 최적화 단계에서 데이터셋을 반복하는 횟수
         하나의 에폭은 다음 두 부분으로 구성됩니다:
         - train loop - 학습용 데이터셋을 반복(iterate)하고 최적의 매개변수로 수렴합니다.
         - validation/test loop - 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복(iterate)합니다.
     - 배치 크기(batch size) - 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
     - 학습률(learning rate) - 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 
    값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있습니다.'''

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    '''손실 함수(loss function)
    일반적인 손실함수에는 regression task에 사용하는 nn.MSELoss 나 classification에 사용하는 nn.NLLLoss,
    그리고 nn.LogSoftmax 와 nn.NLLLoss 를 합친 nn.CrossEntropyLoss등이 있습니다.'''
    loss_fn = nn.CrossEntropyLoss()

    '''옵티마이저(Optimizer)
    최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정입니다.
    모든 최적화 절차는 optimizer 객체에 캡슐화(encapsulate)됩니다. 여기서는 SGD 옵티마이저를 사용하고 있으며, 
    PyTorch에는 ADAM이나 RMSProp과 같은 다른 종류의 모델과 데이터에서 더 잘 동작하는 다양한 옵티마이저가 있습니다.
    https://pytorch.org/docs/stable/optim.html'''

    optimizer = torch.optim.SGD(NN.parameters(), lr=learning_rate)

''' 학습 단계(loop)에서 최적화는 세단계로 이뤄집니다:
    1. optimizer.zero_grad() 를 호출하여 모델 매개변수의 변화도를 재설정합니다. 
    기본적으로 변화도는 더해지기(add up) 때문에 중복 계산을 막기 위해 반복할 때마다 명시적으로 0으로 설정합니다.
    2. loss.backwards() 를 호출하여 예측 손실(prediction loss)을 역전파합니다. PyTorch는 각 매개변수에 대한 손실의 변화도를 저장합니다.
    3. 변화도를 계산한 뒤에는 optimizer.step() 을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정합니다.'''

epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    NN.train_loop(train_dataloader, NN, loss_fn, optimizer)
    NN.test_loop(test_dataloader, NN, loss_fn)

print("Done!")
