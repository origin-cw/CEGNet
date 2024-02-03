import torch
import torch.nn as nn

class EEMBranch(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(EEMBranch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        global_feature = self.avg_pool(x)
        weighted_feature = self.conv1(global_feature) + global_feature
        return self.conv2(weighted_feature)

class EEM(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(EEM, self).__init__()
        self.branch1 = EEMBranch(input_channels, output_channels)
        self.branch2 = EEMBranch(input_channels, output_channels)
        self.conv = nn.Conv2d(output_channels, output_channels, kernel_size=1)

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        fused_feature = self.conv(branch1_output + branch2_output)
        return fused_feature
