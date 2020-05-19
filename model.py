import torch.nn as nn
import torch.nn.functional as F
import torch


class AnomalyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, (11, 11), stride=(1, 1), padding=5)
        self.bn1 = nn.BatchNorm2d(48)

        self.conv2 = nn.Conv2d(48, 48, (9, 9), stride=(2, 2), padding=4)
        self.bn2 = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(48, 48, (7, 7), stride=(2, 2), padding=3)
        self.bn3 = nn.BatchNorm2d(48)

        self.conv4 = nn.Conv2d(48, 48, (5, 5), stride=(2, 2), padding=2)
        self.bn4 = nn.BatchNorm2d(48)

        self.conv5 = nn.Conv2d(48, 48, (3, 3), stride=(2, 2), padding=1)
        self.bn5 = nn.BatchNorm2d(48)

        self.conv_tr1 = nn.ConvTranspose2d(
            48, 48, (5, 5), stride=(2, 2), padding=2, output_padding=1)
        self.bn_tr1 = nn.BatchNorm2d(48)

        self.conv_tr2 = nn.ConvTranspose2d(
            96, 48, (7, 7), stride=(2, 2), padding=3, output_padding=1)
        self.bn_tr2 = nn.BatchNorm2d(48)

        self.conv_tr3 = nn.ConvTranspose2d(
            96, 48, (9, 9), stride=(2, 2), padding=4, output_padding=1)
        self.bn_tr3 = nn.BatchNorm2d(48)

        self.conv_tr4 = nn.ConvTranspose2d(
            96, 48, (11, 11), stride=(2, 2), padding=5, output_padding=1)
        self.bn_tr4 = nn.BatchNorm2d(48)

        self.conv_output = nn.Conv2d(96, 1, (1, 1), (1, 1))
        self.bn_output = nn.BatchNorm2d(1)

    def forward(self, x):
        slope = 0.2
        x = F.leaky_relu((self.bn1(self.conv1(x))), slope)
        x1 = F.leaky_relu((self.bn2(self.conv2(x))), slope)
        x2 = F.leaky_relu((self.bn3(self.conv3(x1))), slope)
        x3 = F.leaky_relu((self.bn4(self.conv4(x2))), slope)
        x4 = F.leaky_relu((self.bn5(self.conv5(x3))), slope)

        x5 = F.leaky_relu(self.bn_tr1(self.conv_tr1(x4)), slope)
        x6 = F.leaky_relu(self.bn_tr2(
            self.conv_tr2(torch.cat([x5, x3], 1))), slope)
        x7 = F.leaky_relu(self.bn_tr3(
            self.conv_tr3(torch.cat([x6, x2], 1))), slope)
        x8 = F.leaky_relu(self.bn_tr4(
            self.conv_tr4(torch.cat([x7, x1], 1))), slope)

        output = F.leaky_relu(self.bn_output(
            self.conv_output(torch.cat([x8, x], 1))), slope)
        return output

if __name__ == "__main__":
    x = torch.rand([16,1,512,512])
    model = AnomalyAE()
    y = model(x)
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)