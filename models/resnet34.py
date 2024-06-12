import torch
import torch.nn as nn
import torch.nn.functional as F
from models import resnet_utils as res_net
# import resnet_utils as res_net

class resnet34(nn.Module):
    def __init__(self, in_channel=6, strides=[2, 2, 1]):
        super(resnet34, self).__init__()
        # mnk modify
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=strides[0], padding=1, bias=False), nn.BatchNorm2d(128))
        self.downsample3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=strides[1], padding=1, bias=False), nn.BatchNorm2d(256))
        self.downsample4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=strides[2], padding=1, bias=False), nn.BatchNorm2d(512))

        self.embedding = nn.Sequential(nn.Conv2d(128 + 512 + 512, 1024, kernel_size=1),nn.BatchNorm2d(1024))  


        self.block1 = nn.Sequential(
            res_net.BasicBlock(64, 64),
            res_net.BasicBlock(64, 64),
            res_net.BasicBlock(64, 64)
        )
        self.block2 = nn.Sequential(
            res_net.BasicBlock(64, 128, stride=strides[0], downsample=self.downsample2),
            res_net.BasicBlock(128, 128),
            res_net.BasicBlock(128, 128),
            res_net.BasicBlock(128, 128)
        )
        self.block3 = nn.Sequential(
            res_net.BasicBlock(128, 256, stride=strides[1], downsample=self.downsample3),
            res_net.BasicBlock(256, 256),
            res_net.BasicBlock(256, 256),
            res_net.BasicBlock(256, 256),
            res_net.BasicBlock(256, 256),
            res_net.BasicBlock(256, 256)
        )
        self.block4 = nn.Sequential(
            res_net.BasicBlock(256, 512, stride=strides[2], downsample=self.downsample4),
            res_net.BasicBlock(512, 512),
            res_net.BasicBlock(512, 512)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        l1 = self.block1(x)
        l2 = self.block2(l1)
        l3 = self.block3(l2)
        l4 = self.block4(l3)
        lg = F.adaptive_avg_pool2d(l4, (1, 1))

        l4 = F.interpolate(l4, l2.size()[-2:], mode='bilinear')
        lg = F.adaptive_avg_pool2d(lg, l2.size()[-2:])

        ft = self.embedding(torch.cat([l2, l4, lg], dim=1))
        return ft

def conv_bn_relu(in_channel, out_channel, kernel_sz=3, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_sz, stride, pad),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class plain_cnn(nn.Module):
    def __init__(self, in_channel=3, strides=[2, 2, 1]):
        super(plain_cnn, self).__init__()

        self.block1 = nn.Sequential(
            conv_bn_relu(in_channel, 64, 3, 1, 1),
            conv_bn_relu(64, 64, 3, 1, 1),
            conv_bn_relu(64, 64, 3, 1, 1),
            conv_bn_relu(64, 64, 3, 1, 1),
            conv_bn_relu(64, 64, 3, 1, 1),
        )
        self.block2 = nn.Sequential(
            conv_bn_relu(64, 128, 3, 2, 1),
            conv_bn_relu(128, 128, 3, 1, 1),
            conv_bn_relu(128, 128, 3, 1, 1),
            conv_bn_relu(128, 128, 3, 1, 1),
            conv_bn_relu(128, 128, 3, 1, 1),
        )
        self.block3 = nn.Sequential(
            conv_bn_relu(128, 256, 3, 2, 1),
            conv_bn_relu(256, 256, 3, 1, 1),
            conv_bn_relu(256, 256, 3, 1, 1),
            conv_bn_relu(256, 256, 3, 1, 1),
            conv_bn_relu(256, 256, 3, 1, 1),
        )

        self.block4 = nn.Sequential(
            conv_bn_relu(256, 512, 3, 2, 1),
            conv_bn_relu(512, 512, 3, 1, 1),
        )

        self.embedding = nn.Sequential(
            nn.Conv2d(256 + 512, 512, kernel_size=1),
            nn.BatchNorm2d(512)
        )

    def forward(self, x):
        l1 = self.block1(x)
        l2 = self.block2(l1)
        l3 = self.block3(l2)
        l4 = self.block4(l3)
        lg = F.adaptive_avg_pool2d(l4, (1, 1))
        lg = F.adaptive_avg_pool2d(lg, l3.size()[-2:])

        ft = self.embedding(torch.cat([l3, lg], dim=1))
        return ft
    

# # if __name__ == "__main__":  
# # 假设输入张量的形状为(batch_size, in_channels, height, width)
# if __name__ == "__main__":
#     # 假设输入张量的形状为(batch_size, in_channels, height, width)
#     input_tensor = torch.randn(8, 3, 540, 720)  # batch_size=8, in_channels=3, height=540, width=720
#     model = resnet34(in_channel=3, strides=[2, 2, 1])
#     output = model(input_tensor)
#     print(output.shape)  # 打印输出张量的形状


