import torch
import torch.nn as nn
import torch.nn.functional as F
from THNet_Distillation.BAFNet.SSA import shunted_t
from torch.nn.functional import interpolate


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, padding, dilation):
        super(DepthWiseConv, self).__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=in_channel,
                      kernel_size=3,
                      stride=1,
                      padding=padding,
                      dilation=dilation,
                      bias=False,
                      groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False,
                      groups=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Up(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Up, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


class BYS(torch.nn.Module):
    def __init__(self, in_plane1):
        super(BYS, self).__init__()

        self.weight_mu = nn.Parameter(torch.Tensor(in_plane1, in_plane1, 1, 1).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(in_plane1, in_plane1, 1, 1).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(in_plane1).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(in_plane1).uniform_(-5, -4))
        self.weight_epsilon = nn.Parameter(
            torch.distributions.laplace.Laplace(0, 1).sample((in_plane1, in_plane1, 1, 1)))
        self.bias_epsilon = nn.Parameter(torch.distributions.laplace.Laplace(0, 1).sample((in_plane1,)))

    def forward(self, rgb_t):

        weight_std = F.softplus(self.weight_rho)
        weight = self.weight_mu + weight_std * self.weight_epsilon

        bias_std = F.softplus(self.bias_rho)
        bias = self.bias_mu + bias_std * self.bias_epsilon

        out = F.conv2d(rgb_t, weight, bias)

        return out


class PGI(nn.Module):
    def __init__(self, in_channels):
        super(PGI, self).__init__()

        self.bys = BYS(in_channels)

        self.query_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.con_k = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, rgb):

        rgb = self.bys(rgb)

        query = self.query_convs(rgb)
        channel_attention = self.channel_attention(query)
        spatial_attention = self.spatial_attention(query)
        attention = channel_attention * spatial_attention

        key = self.con_k(torch.max(rgb, dim=1, keepdim=True)[0]) * rgb
        key = key * attention + query

        return key


class ADR(torch.nn.Module):
    def __init__(self, in_plane1):
        super(ADR, self).__init__()

        self.bys = BYS(in_plane1)

        self.con_v1 = DepthWiseConv(in_plane1, in_plane1 // 4, padding=1, dilation=1)
        self.con_v2 = DepthWiseConv(in_plane1, in_plane1 // 4, padding=3, dilation=3)
        self.con_v3 = DepthWiseConv(in_plane1, in_plane1 // 4, padding=5, dilation=5)
        self.con_v4 = DepthWiseConv(in_plane1, in_plane1 // 4, padding=7, dilation=7)

    def forward(self, t):

        out_t = self.bys(t)

        t = out_t
        value1 = self.con_v1(t)
        value2 = self.con_v2(t)
        value3 = self.con_v3(t)
        value4 = self.con_v4(t)
        out_t = torch.cat((value1, value2, value3, value4), dim=1)

        attention_w = torch.sigmoid(out_t)
        value = attention_w / torch.sum(attention_w, dim=(2, 3), keepdim=True)

        return value


class Decoder(nn.Module):
    def __init__(self, in1, in2):
        super(Decoder, self).__init__()
        self.bcon1 = BasicConv2d(in_planes=in1, out_planes=in1, kernel_size=1, stride=1, padding=0)
        self.bcon2 = BasicConv2d(in_planes=in2, out_planes=in1, kernel_size=1, stride=1, padding=0)
        self.bconv = BasicConv2d(in_planes=in1 * 2, out_planes=in1, kernel_size=1, stride=1, padding=0)
        self.upsample2 = Up(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, d1, d2):
        d1 = self.bcon1(d1)
        d2 = self.bcon2(d2)
        d2 = self.upsample2(d2)
        out = torch.cat((d1, d2), dim=1)
        out = self.bconv(out)
        return out


class Net_S(nn.Module):
    def __init__(self, ):
        super(Net_S, self).__init__()
        self.backbone = shunted_t()
        load_state_dict = torch.load(
            '/media/hjk/shuju/THNet_Distillation/THNet_Distillation/Net_pth/ckpt_T.pth')
        self.backbone.load_state_dict(load_state_dict)
        self.conv1_3 = BasicConv2d(1, 3, kernel_size=3, stride=1, padding=1)

        self.bqk1 = PGI(64)
        self.bqk2 = PGI(128)
        self.bqk3 = PGI(256)
        self.bqk4 = PGI(512)

        self.bv1 = ADR(64)
        self.bv2 = ADR(128)
        self.bv3 = ADR(256)
        self.bv4 = ADR512)


        self.decoder1 = Decoder(in1=256, in2=512)
        self.decoder2 = Decoder(in1=128, in2=256)
        self.decoder3 = Decoder(in1=64, in2=128)

        self.last1 = nn.Sequential(
            Up(scale_factor=16, mode='bilinear', align_corners=True),
            BasicConv2d(256, 1, 3, 1, 1)
        )
        self.last2 = nn.Sequential(
            Up(scale_factor=8, mode='bilinear', align_corners=True),
            BasicConv2d(128, 1, 3, 1, 1)
        )
        self.last3 = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True),
            BasicConv2d(64, 1, 3, 1, 1)
        )

    def forward(self, x, y):
        merges = []
        rgb = self.backbone(x)
        t = self.backbone(self.conv1_3(y))

        qk1 = self.bqk1(rgb[0])
        qk2 = self.bqk2(rgb[1])
        qk3 = self.bqk3(rgb[2])
        qk4 = self.bqk4(rgb[3])


        v1 = self.bv1(t[0])
        v2 = self.bv2(t[1])
        v3 = self.bv3(t[2])
        v4 = self.bv4(t[3])

        merges.append(qk1 * v1)
        merges.append(qk2 * v2)
        merges.append(qk3 * v3)
        merges.append(qk4 * v4)

        out1 = self.decoder1(merges[2], merges[3])
        out2 = self.decoder2(merges[1], out1)
        out3 = self.decoder3(merges[0], out2)

        return self.last3(out3), self.last2(out2), self.last1(out1), merges[0], merges[1], merges[2], merges[3]


if __name__ == '__main__':
    model = Net_S().cuda()
    left = torch.randn(2, 3, 256, 256).cuda()
    right = torch.randn(2, 1, 256, 256).cuda()
    # out = model(left, right)
    out = model(left, right)

    # from thop import profile
    # flops, params = profile(model, (left, right))
    # print("flops:", flops/10000000)
    # print("params:", params/1000000)

    # from ptflops import get_model_complexity_info
    # flops , params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
    # print('Flops:', flops)
    # print('Params:', params)

    print("==> Total params: % .2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    for i in out:
        print(i.shape)
