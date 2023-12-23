
from PIL import Image

from codes.ETC._mae import *
from codes.ETC._vit import *
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

autogrades = False
if autogrades:
    # 신경망을 학습할 때 가장 자주 사용되는 알고리즘은 역전파 입니다. 이 알고리즘에서,
    # 모델 가중치는 주어진 매개변수에 대한 손실 함수의 gradient 에 따라 조정됩니다.
    # 이러한 변화도를 계산하기 위해 PyTorch에는 autograd 라고 불리는 자동 미분 엔진이 내장되어 있습니다.

    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")

    # 신경망에서 매개변수의 가중치를 최적화하려면 매개변수에 대한 손실함수의 도함수(derivative)를 계산해야 합니다.
    # 이러한 도함수를 계산하기 위해, `loss.backward()` 를 호출한 다음 `w.grad` 와 `b.grad` 에서 값을 가져옵니다.
    # 만약 동일한 그래프에서 여러번의 `backward` 호출이 필요하면,`backward` 호출 시에 `retrain_graph=True` 를 전달해야 합니다.

    loss.backward()
    print(w.grad)
    print(b.grad)

    # requires_grad=True 인 모든 텐서들은 연산 기록을 추적하고 변화도 계산을 지원합니다.
    # 그러나 모델을 학습한 뒤 입력 데이터를 단순히 적용하기만 하는 경우와 같이 순전파 연산만 필요한 경우 필요 없을 수 있습니다.
    # 연산 코드를 ``torch.no_grad()`` 블록으로 둘러싸서 연산 추적을 멈출 수 있습니다.
    # 변화도 추적을 멈춰야 하는 이유들은 다음과 같습니다:
    # - 신경망의 일부 매개변수를 고정된 매개변수(frozen parameter) 로 표시합니다.
    # - 변화도를 추적하지 않는 텐서의 연산이 더 효율적이기 때문에, 순전파 단계만 수행할 때 연산 속도가 향상됩니다.

    z = torch.matmul(x, w) + b
    print(z.requires_grad)

    with torch.no_grad():
        z = torch.matmul(x, w) + b
    print(z.requires_grad)

    z = torch.matmul(x, w) + b
    z_det = z.detach()
    print(z_det.requires_grad)

    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp + 1).pow(2).t()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"First call\n{inp.grad}")
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nSecond call\n{inp.grad}")
    inp.grad.zero_()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{inp.grad}")

    # 순전파 단계
    # - 요청된 연산을 수행하여 결과 텐서를 계산.
    # - DAG에 연산의 gradient function 를 유지.
    #
    # 역전파 단계(root에서 .backward() 가 호출될 때 시작)
    # - 각 .grad_fn 으로부터 변화도를 계산.
    # - 각 텐서의 .grad 속성에 계산 결과를 accumulate.
    # - 연쇄 법칙을 사용하여, 모든 잎(leaf) 텐서들까지 전파(propagate).

    # 출력 함수가 임의의 텐서인 경우가 있습니다. 이럴 때, PyTorch는 실제 변화도가 아닌 야코비안 곱(Jacobian product) 을 계산.
    # 동일한 인자로 `backward` 를 두차례 호출하면 변화도 값이 달라집니다.
    # 이는 ``역방향`` 전파를 수행할 때, PyTorch가 변화도를 accumulate하기 때문입니다.
    # 따라서 제대로 된 변화도를 계산하기 위해서는 grad 속성을 먼저 0으로 만들어야 합니다.
    # 실제 학습 과정에서는 *옵티마이저(optimizer)*\ 가 이 과정을 도와줍니다.

model = False
if model:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    '''신경망은 데이터에 대한 연산을 수행하는 계층(layer)/모듈(module)로 구성되어 있습니다.
    PyTorch의 모든 모듈은 nn 의 하위 클래스입니다. 신경망은 다른 모듈(계층; layer)로 구성된 모듈입니다.
    신경망 모델을 `nn.Module` 의 하위클래스로 정의하고, ``__init__`` 에서 신경망 계층들을 초기화합니다.
    `nn.Module` 을 상속받은 모든 클래스는 `forward` 메소드에 입력 데이터에 대한 연산들을 구현합니다.'''


    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        # `model.forward()` 를 직접 호출하지 마세요!
        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


    if True:
        model = NeuralNetwork().to(device)
        print(model)

        X = torch.rand(1, 28, 28, device=device)
        logits = model(X)
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)
        print(f"Predicted class: {y_pred}")

        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    # Example
    # ## 모델 계층(Layer)
    # input_image = torch.rand(3, 28, 28)
    #
    # # nn.Flatten 는 계층을 초기화하여 각 28x28의 2D 이미지를 784 픽셀 값을 갖는 연속된 배열로 변환합니다.
    # flatten = nn.Flatten()
    # flat_image = flatten(input_image)
    #
    # # 선형 계층은 저장된 가중치(weight)와 편향(bias)을 사용하여 입력에 선형 변환(linear transformation)을 적용하는 모듈입니다.
    # layer1 = nn.Linear(in_features=28 * 28, out_features=20)
    # hidden1 = layer1(flat_image)
    #
    # # 비선형 활성화(activation)는 모델의 입력과 출력 사이에 복잡한 관계(mapping)를 만듭니다.
    # # 비선형 활성화는 선형 변환 후에 적용되어 비선형성을 도입하고, 신경망이 다양한 현상을 학습할 수 있도록 돕습니다.
    # # 이 모델에서는 nn.ReLU 를 선형 계층들 사이에 사용하지만, 모델을 만들 때는 비선형성을 가진 다른 활성화를 도입할 수도 있습니다.
    # print(f"Before ReLU: {hidden1}\n\n")
    # hidden1 = nn.ReLU()(hidden1)
    # print(f"After ReLU: {hidden1}")
    #
    # # nn.Sequential 은 순서를 갖는 모듈의 컨테이너입니다. 데이터는 정의된 것과 같은 순서로 모든 모듈들을 통해 전달됩니다.
    # seq_modules = nn.Sequential(
    #     flatten,
    #     layer1,
    #     nn.ReLU(),
    #     nn.Linear(20, 10)
    # )
    # input_image = torch.rand(3, 28, 28)
    # logits = seq_modules(input_image)
    #
    # # 신경망의 마지막 선형 계층은 nn.Softmax 모듈에 전달될 logits 는 모델의 각 분류(class)에 대한 예측 확률을 나타내도록
    # # [0, 1] 범위로 비례하여 조정(scale)됩니다. `dim` 매개변수는 값의 합이 1이 되는 차원을 나타냅니다.
    # softmax = nn.Softmax(dim=1)
    # pred_probab = softmax(logits)

    # 신경망 내부의 많은 계층들은 *매개변수화(parameterize)* 됩니다. 즉, 학습 중에 최적화되는 가중치와 편향과 연관지어집니다.
    # `nn.Module` 을 상속하면 모델 객체 내부의 모든 필드들이 자동으로 추적(track)되며, 모델의 ``parameters()`` 및
    # `named_parameters()` 메소드로 모든 매개변수에 접근할 수 있게 됩니다.
    # 이 예제에서는 각 매개변수들을 순회하며(iterate), 매개변수의 크기와 값을 출력합니다.

save = False
if save:
    model = models.vgg16(weights='IMAGENET1K_V1')
    torch.save(model.state_dict(), 'model_weights.pth')

    '''모델 가중치를 불러오기 위해서는, 먼저 동일한 모델의 인스턴스(instance)를 생성한 다음에 
    load_state_dict() 메소드를 사용하여 매개변수들을 불러옵니다.'''
    model = models.vgg16()  # 여기서는 ``weights`` 를 지정하지 않았으므로, 학습되지 않은 모델을 생성합니다.
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

    torch.save(model, 'model.pth')

    model = torch.load('model.pth')

masking = False
if masking:
    # Define the custom dataset for training (with both train and masks)
    class YourSatelliteDatasetWithMask(YourSatelliteDataset):
        def __getitem__(self, idx):
            data = super().__getitem__(idx)
            # Assuming you have a method to generate masks for your train
            mask = generate_mask(data['image'])
            return {'image': data['image'], 'mask': mask}


    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate your transfer learning model
    transfer_model = YourTransferLearningModel()
    transfer_model.to(device)

    # Instantiate your dataset class with masks
    train_dataset = YourSatelliteDatasetWithMask(root='/path/to/train/train', transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Define your masking model (simple example using a binary classification)
    masking_model = nn.Sequential(
        nn.Conv2d(in_channels=your_num_channels, out_channels=1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )
    masking_model.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(masking_model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        masking_model.train()
        for batch in train_dataloader:
            images, masks = batch['image'].to(device), batch['mask'].to(device)

            # Forward pass
            predictions = masking_model(images)
            loss = criterion(predictions, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained masking model if needed
    torch.save(masking_model.state_dict(), 'trained_masking_model.pth')

masking2 = False
if masking2:
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Assuming your transfer learning model is already defined and trained
    transfer_model = YourTransferLearningModel()
    transfer_model.to(device)
    transfer_model.eval()

    # Assuming your dataset class is already defined
    dataset = YourSatelliteDataset(root='/path/to/satellite/train', transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    # Function to perform masking
    def perform_masking(model, dataloader, threshold=0.5):
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)

                # Assuming the transfer learning model produces segmentation masks
                segmentation_masks = model(images)

                # Applying a threshold to obtain binary masks (building vs. background)
                binary_masks = (segmentation_masks > threshold).float()

                # Perform any additional post-processing if needed

                # Now binary_masks contain the building masks for the train in the batch
                # You can use these masks for further processing or visualization
                # For example, you might want to save the masks as train or use them in other tasks


    # Example usage
    perform_masking(transfer_model, dataloader)

mae = True
if mae:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load a sample image
    image_path = "../../database/PNGImages/austin1.png"  # Replace with the actual path to your image
    image = Image.open(image_path)

    # Preprocess the image (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    input_image = input_image.to(device)

    image_size = 5000
    patch_size = 5000
    num_classes = 2
    dim = 512
    depth = 12
    heads = 8
    mlp_dim = 1024
    pool = 'cls'
    channels = 3
    dim_head = 64
    dropout = 0.1
    emb_dropout = 0.1

    vit_model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        pool=pool,
        channels=channels,
        dim_head=dim_head,
        dropout=dropout,
        emb_dropout=emb_dropout
    )

    vit_model = vit_model.to(device)

    # Instantiate the MAE model (make sure it's already trained or loaded with trained weights)
    mae_model = MAE(encoder=vit_model, decoder_dim=5000, masking_ratio=0.75, decoder_depth=1, decoder_heads=8,
                    decoder_dim_head=64)

    mae_model = mae_model.to(device)

    # Set the model to evaluation mode
    mae_model.eval()

    # Forward pass to generate the masked image
    with torch.no_grad():
        recon_loss = mae_model(input_image)

    # Get the masked image from the input image
    masked_image = mae_model.to_pixels(mae_model.mask_token)

    def mask_test_image(image_path, model):
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        model=model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(image)

        mask = torch.sigmoid(output[0, 0]).numpy()

        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.show()

    # Display the original and masked train
    original_image_np = input_image.cpu().squeeze().permute(1, 2, 0).numpy()


    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image_np)
    plt.axis('off')



    plt.show()

masking3 = False
if masking3:
    class CustomImageDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.image_paths = [os.path.join(data_dir, 'PNGImages', filename) for filename in
                                os.listdir(os.path.join(data_dir, 'PNGImages'))]
            self.mask_paths = [os.path.join(data_dir, 'PedMasks', filename) for filename in
                               os.listdir(os.path.join(data_dir, 'PedMasks'))]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            mask = Image.open(self.mask_paths[idx])

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            sample = {"image": image, "mask": mask}
            return sample


    # Example usage
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomImageDataset(data_dir='../database', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)



    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)



    # 모델 정의 및 전이 학습
    model = models.resnet18(pretrained=True, progress=True)

    model=model.to(device)
    # 모델의 마지막 레이어를 세그멘테이션에 맞게 조정
    model.fc = nn.Conv2d(512, 1, kernel_size=1)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # 모델 학습 (여기서는 간략하게 훈련 반복을 보여줍니다)
    # for epoch in range(5):  # 예시로 5 에폭만 수행합니다
    #     for images, masks in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, masks)
    #         loss.backward()
    #         optimizer.step()


    # 테스트 이미지 마스킹 함수 정의
    def mask_test_image(image_path, model):
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(image)

        mask = torch.sigmoid(output[0, 0]).numpy()

        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.show()


    def mask_test_images(folder_path, model, transform):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                mask_test_image(image_path, model, transform)


    def mask_test_image(image_path, model, transform):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image)

        mask = torch.sigmoid(output[0, 0]).cpu().numpy()

        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.show()


    # Example usage
    test_image_path = '../database/test'  # 테스트 이미지 파일 경로
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    mask_test_images(test_image_path, model, transform)

data_lader=False
if data_lader:
    import os
    import torch
    import pandas as pd
    import matplotlib.pyplot as plt
    from torchvision import datasets
    from torch.utils.data import Dataset
    from torchvision.io import read_image
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor, Lambda


    class CustomImageDataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
            self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

            sample = {"image": image, "label": label}
            return sample


    if __name__ == '__main__':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)

        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
            target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
            # 이 함수는 먼저 (데이터셋 정답의 개수인) 크기 10짜리 영 텐서(zero tensor)를 만들고,
            # scatter 를 호출하여 주어진 정답 `y` 에 해당하는 인덱스에 `value=1` 을 할당.
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        labels_map = {
            0: "T-Shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }

        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(training_data), size=(1,)).item()
            img, label = training_data[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title(labels_map[label])
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
        plt.show()

        # 아래의 각 순회(iteration)는 (각각 ``batch_size=64`` 의 특징(feature)과 정답(label)을 포함하는) `train_features` 와
        # `train_labels` 의 묶음(batch)을 반환합니다. `shuffle=True` 로 지정했으므로, 모든 배치를 순회한 뒤 데이터가 섞입니다.
        train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

        train_features, train_labels = next(iter(train_dataloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0].squeeze()
        label = train_labels[0]
        plt.imshow(img, cmap="gray")
        plt.show()
        print(f"Label: {label}")

    '''
    1. dataset 과 model learning code 는 분리 하는 것이 이상적.
    2. PyTorch 는 `torch.utils.data.DataLoader` 와 `torch.utils.data.Dataset`를 통해 다양한 dataset 제공.
    3. Dataset 은 sample 과 ground truth 를 저장하고, DataLoader 는 Dataset 을 쉽게 접근 할 수 있도록 iterable 한 객체로 변형.
    4. 모델을 학습할 때, 샘플들을 minibatch 로 전달하고, 매 epoch마다 데이터를 다시 섞어서 과적합(overfit)을 막고, 
    `multiprocessing` 을 사용하여 데이터 검색 속도를 높임.
    5. Data is not always provided to fit on ML. Transform 을 해서 데이터를 조작하고 학습에 적합.
    6. feature(image) 를 변경하기 위한 `transform` 과 label(integer) 을 변경하기 위한 `target_transform` 를 매개변수로 가짐.
    7. 정규화된 텐서 feature와 one-hot으로 encode 된 텐서 형태의 label 필요. -> ToTensor 와 Lambda 사용.

    etc
     - ``root`` 는 학습/테스트 데이터가 저장되는 경로입니다.
     - ``train`` 은 학습용 또는 테스트용 데이터셋 여부를 지정합니다.
     - ``download=True`` 는 ``root`` 에 데이터가 없는 경우 인터넷에서 다운로드합니다.
     - ``transform`` 과 ``target_transform`` 은 특징(feature)과 정답(label) 변형(transform)을 지정합니다.
    '''
