import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# =========================
# Device 설정
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# =========================
# 데이터 전처리
# =========================

train_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3
    ),

    transforms.GaussianBlur(3),

    transforms.ToTensor()
])

val_transform = transforms.Compose([

    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# =========================
# 데이터셋 불러오기
# =========================

train_dataset = ImageFolder(
    root='dataset/train',
    transform=train_transform
)

val_dataset = ImageFolder(
    root='dataset/val',
    transform=val_transform
)

# 클래스 확인
print(train_dataset.classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# =========================
# CNN 모델
# =========================

class DiceCNN(nn.Module):

    def __init__(self):
        super(DiceCNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, 14)
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.fc(x)

        return x

model = DiceCNN().to(device)

# =========================
# Loss / Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# 학습
# =========================

num_epochs = 30

for epoch in range(num_epochs):

    # -------------------------
    # Train
    # -------------------------

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # gradient 초기화
        optimizer.zero_grad()

        # forward
        outputs = model(images)

        # loss 계산
        loss = criterion(outputs, labels)

        # backward
        loss.backward()

        # weight 업데이트
        optimizer.step()

        train_loss += loss.item()

        # accuracy 계산
        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

    train_acc = 100 * train_correct / train_total

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_acc = 100 * val_correct / val_total

    # -------------------------
    # 출력
    # -------------------------

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

# =========================
# 모델 저장
# =========================

torch.save(
    model.state_dict(),
    "dice_cnn.pth"
)

print("Model saved")
