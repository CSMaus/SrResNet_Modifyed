import torch
import torch.nn as nn
import math


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output
class _NetGS(nn.Module):
    def __init__(self, num_blocks):
        super(_NetGS, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, num_blocks)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out




class _Residual_BlockSmall(nn.Module):
    def __init__(self, num_channels):
        super(_Residual_BlockSmall, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(num_channels*2, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class _Residual_BlockBottleneck(nn.Module):
    def __init__(self, num_channels):
        super(_Residual_BlockBottleneck, self).__init__()

        self.conv1x1_reduce = nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x1_expand = nn.Conv2d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = nn.InstanceNorm2d(num_channels, affine=True)
        self.bn2 = nn.InstanceNorm2d(num_channels, affine=True)
        self.bn3 = nn.InstanceNorm2d(num_channels*2, affine=True)

        self.relu = nn.LeakyReLU(0.2, inplace=True)


class _Residual_BlockBottleneckEfficient(nn.Module):
    def __init__(self, num_channels):
        super(_Residual_BlockBottleneckEfficient, self).__init__()

        self.conv1x1_reduce = nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x1_expand = nn.Conv2d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=1, stride=1, padding=0, bias=False)

        # self.bn1 = nn.InstanceNorm2d(num_channels, affine=True)
        # self.bn2 = nn.InstanceNorm2d(num_channels, affine=True)
        self.bn3 = nn.InstanceNorm2d(num_channels*2, affine=True)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1x1_reduce)
        out = self.relu(self.conv3x3(out))
        out = self.bn3(self.conv1x1_expand(out))

        # out = self.relu(self.bn1(self.conv1x1_reduce(x)))
        # out = self.relu(self.bn2(self.conv3x3(out)))
        # out = self.bn3(self.conv1x1_expand(out))
        # out += identity  # Skip connection
        out = torch.add(out, identity)
        return out



class _NetX2(nn.Module):
    def __init__(self, num_blocks, num_channels):
        super(_NetX2, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=num_channels*2, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_BlockBottleneck, num_blocks, num_channels)
        # self.residual = self.make_layer(_Residual_BlockSmall, num_blocks, 64)

        self.conv_mid = nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(num_channels*2, affine=True)

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=num_channels*2, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer, num_channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        return out
class _NetX2Eff(nn.Module):
    def __init__(self, num_blocks, num_channels):
        super(_NetX2Eff, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=num_channels*2, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_BlockBottleneck, num_blocks, num_channels)
        self.residual = self.make_layer(_Residual_BlockSmall, num_blocks*2, num_channels)

        self.conv_mid = nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(num_channels*2, affine=True)

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=num_channels*2, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer, num_channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        return out

class _NetG(nn.Module):
    def __init__(self):
        super(_NetG, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        return out


