# Builds and trains SmallCNN
from torch import nn


class CatDogCNN(nn.Module):
    # Four VGG-style blocks, then global average pooling, dropout (0.3) and a linear layer
    # This model will work with all input images where w and h are multiples of 16

    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        # Block 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)

        # Classifier: 256 channel averages -> 2 class scores
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

        # ReLU has no weights, so one is reused after every batch norm
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):                           # x: N x 3 x 128 x 128
        # Block 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))  # N x 32 x 128 x 128
        x = self.relu(self.bn1_2(self.conv1_2(x)))  # N x 32 x 128 x 128
        x = self.pool1(x)                           # N x 32 x 64 x 64
        
        # Block 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))  # N x 64 x 64 x 64
        x = self.relu(self.bn2_2(self.conv2_2(x)))  # N x 64 x 64 x 64
        x = self.pool2(x)                           # N x 64 x 32 x 32

        # Block 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))  # N x 128 x 32 x 32
        x = self.relu(self.bn3_2(self.conv3_2(x)))  # N x 128 x 32 x 32
        x = self.pool3(x)                           # N x 128 x 16 x 16

        # Block 4
        x = self.relu(self.bn4_1(self.conv4_1(x)))  # N x 256 x 16 x 16
        x = self.relu(self.bn4_2(self.conv4_2(x)))  # N x 256 x 16 x 16
        x = self.pool4(x)                           # N x 256 x 8 x 8
        
        # Classifier
        x = x.mean(dim=(2, 3))                      # N x 256: global average pooling
        x = self.dropout(x)
        return self.classifier(x)                   # N x 2 class scores (logits)


def init_weights(model):
    # Kaiming initialisation for our network which uses ReLU
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
