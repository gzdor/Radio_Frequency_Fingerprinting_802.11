import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import torch.nn.functional as F


class OracleClassifier(pl.LightningModule):
    '''
    Paper: https://genesys-lab.org/papers/ORACLE-INFOCOM-19.pdf

    The first convolution layer consists of 50 filters, each of size 1 × 7 
    The second convolution layer has 50 filters each of size 2 × 7 

    Each convolution layer is followed by a Rectified Linear Unit (ReLU)

    The first fully connected layer has 256 neurons
    The second fully connected layer of 80 neurons

    Set the dropout rate to 50% at the dense layers

    A softmax classifier is used in the last layer to output the 
      probabilities of each sample being fed to the CNN

    L2 regularization parameter λ = 0.0001.
    Trained using Adam optimizer with a learning rate=0.0001

    '''
    def __init__(self, dropout=0.5, lr=0.0001, num_classes=16):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(2, 50, kernel_size=7)
        self.conv2 = nn.Conv1d(50, 50, kernel_size=7)
        self.fc1 = nn.Linear(50 * (128 - 6 - 6), 256)
        self.fc2 = nn.Linear(256, 80)
        self.fc3 = nn.Linear(80, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.save_hyperparameters()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
    def _common_step(self, batch, type):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y.squeeze())
        acc = accuracy(
            torch.argmax(logits, dim=1), 
            y.squeeze(),
            task='multiclass',
            num_classes=self.num_classes
        )
        self.log(f'{type}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{type}_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss 

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "validation")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

class CNNClassifier(pl.LightningModule):
    def __init__(self, lr=0.0001, dropout=0.15, num_classes=16):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * 128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.save_hyperparameters()

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def _common_step(self, batch, type):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y.squeeze())
        acc = accuracy(
            torch.argmax(logits, dim=1), 
            y.squeeze(),
            task='multiclass',
            num_classes=self.num_classes
        )
        self.log(f'{type}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{type}_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "validation")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out

class ResCNN(pl.LightningModule):
    def __init__(self, lr, dropout, num_classes=16):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.res_block = ResidualBlock(32, 64)
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * 128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.save_hyperparameters()

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x))) 
        x = self.res_block(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def _common_step(self, batch, type):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y.squeeze())
        acc = accuracy(
            torch.argmax(logits, dim=1), 
            y.squeeze(),
            task='multiclass',
            num_classes=self.num_classes
        )
        self.log(f'{type}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{type}_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss 

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "validation")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

import torchvision.models as models

class EfficientNetTL(pl.LightningModule):
    def __init__(self, num_classes=16):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv = nn.Conv2d(2, 3, kernel_size=1, stride=1, padding=0, bias=False)
        # self.pool = nn.AdaptiveAvgPool2d((32, 32))
        self.pool = nn.AdaptiveAvgPool2d((128, 128))

        backbone = models.efficientnet_b0(weights="DEFAULT")
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
                            nn.Dropout(p=0.2, inplace=True),
                            nn.Linear(in_features=1280, out_features=self.num_classes, bias=True)
        )

    def forward(self, x):
        x = x.unsqueeze(-2)  # Add an extra dimension to match the expected 4D input
        x = self.conv(x)
        x = self.pool(x)
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x
   
    def _step(self, batch, batch_idx, step_type):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y.squeeze())
        acc = accuracy(
            torch.argmax(logits, dim=1), 
            y.squeeze(),
            task='multiclass',
            num_classes=self.num_classes
        )
        self.log(f'{step_type}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{step_type}_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    

# class EfficientNetTL2(pl.LightningModule):
#     def __init__(self, num_classes=16):
#         super().__init__()
#         self.num_classes = num_classes
        
#         self.conv = nn.Conv2d(2, 3, kernel_size=1, stride=1, padding=0, bias=False)
#         # self.pool = nn.AdaptiveAvgPool2d((32, 32))
#         self.pool = nn.AdaptiveAvgPool2d((128, 128))

#         backbone = models.efficientnet_b0(weights="DEFAULT")
#         layers = list(backbone.children())[:-1]
#         self.feature_extractor = nn.Sequential(*layers)
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
#         self.classifier = nn.Sequential(
#                             nn.Dropout(p=0.2, inplace=True),
#                             nn.Linear(in_features=1280, out_features=self.num_classes, bias=True)
#         )

#     def forward(self, x):
#         x = x.unsqueeze(-2)  # Add an extra dimension to match the expected 4D input
#         x = self.conv(x)
#         x = self.pool(x)
#         representations = self.feature_extractor(x).flatten(1)
#         x = self.classifier(representations)
#         return x
   
#     def _step(self, batch, batch_idx, step_type):
#         x, y = batch
#         logits = self(x)
#         loss = nn.CrossEntropyLoss()(logits, y.squeeze())
#         acc = accuracy(
#             torch.argmax(logits, dim=1), 
#             y.squeeze(),
#             task='multiclass',
#             num_classes=self.num_classes
#         )
#         self.log(f'{step_type}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log(f'{step_type}_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         return loss

#     def training_step(self, batch, batch_idx):
#         return self._step(batch, batch_idx, "train")

#     def validation_step(self, batch, batch_idx):
#         return self._step(batch, batch_idx, "val")

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
#         return optimizer
    

class ResNet50TL(pl.LightningModule):
    def __init__(self, num_classes=16):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv = nn.Conv2d(2, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((128, 128))

        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, self.num_classes)

    def forward(self, x):
        x = x.unsqueeze(-2)  # Add an extra dimension to match the expected 4D input
        x = self.conv(x)
        x = self.pool(x)
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x
   
    def _step(self, batch, batch_idx, step_type):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y.squeeze())
        acc = accuracy(
            torch.argmax(logits, dim=1), 
            y.squeeze(),
            task='multiclass',
            num_classes=self.num_classes
        )
        self.log(f'{step_type}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{step_type}_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer