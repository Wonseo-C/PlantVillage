# Shallow CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()

        # vgg16 structure
        # Input (256, 256, 3)
        self.features = nn.Sequential(
            # Conv 1-1, 1-2 (256, 256, 64)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),

            # Max Pool (128, 128, 64)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # Conv 2-1, 2-2 (128, 128, 128)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        # Global Average Pool (128,)
        self.globalAveragePool = nn.AdaptiveAvgPool2d(1)


    def forward(self, inputImage):
        vggOutput = self.features(inputImage)
        output = self.globalAveragePool(vggOutput)

        return torch.squeeze(output)

class ShallowVGG(nn.Module):
    def __init__(self, dropout=0.9):
        super(ShallowVGG, self).__init__()

        # vgg16 structure
        # Input (256, 256, 3)
        self.features = nn.Sequential(
            # Conv 1-1, 1-2 (256, 256, 64)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),

            # Max Pool (128, 128, 64)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # Conv 2-1, 2-2 (128, 128, 128)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        # Global Average Pool (128,)
        self.globalAveragePool = nn.AdaptiveAvgPool2d(1)

        # Instead of PCA and (Random Forest, SVM)
        # Fully connect layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=39),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        


    def forward(self, inputImage):
        feature1 = self.features(inputImage)
        feature2 = self.globalAveragePool(feature1)
        output = self.fc(torch.reshape(feature2, (1, -1)))
        
        return F.softmax(output, dim=1)