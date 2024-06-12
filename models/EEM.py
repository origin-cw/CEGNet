import torch
import torch.nn as nn
import torch.nn.functional as F

class EEM(nn.Module):
    def __init__(self):
        super(EEM, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=4)
        self.max_pool = nn.MaxPool2d(kernel_size=32)

        self.branch1_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid()
        )

        self.branch2_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=2,padding=2),
            nn.BatchNorm2d(256),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.conv_layers(x)

        ft = self.avg_pool(x)
        max_ft = self.max_pool(ft)

        branch1 = self.branch1_conv(ft)
        branch2 = self.branch2_conv(ft)

        branch1_out = torch.mul(branch1, max_ft)
        branch2_out = torch.mul(branch2, max_ft)

        branch2_out_repeat = branch2_out.repeat(1, 1, 2, 2)

        mixed_feature = torch.cat([branch1_out ,branch2_out_repeat], dim=1)

        return mixed_feature
    