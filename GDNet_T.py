import torch
import torch.nn as nn
import torch.nn.functional as F
from THNet_Distillation.BAFNet.SSA import shunted_b
from THNet_Distillation.BAFNet.GCN import GloRe_Unit_2D
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

#PGI
class PGI(nn.Module):
    def __init__(self, in_channels):
        super(PGI, self).__init__()

        self.bys = BYS(in_channels)

        self.query_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
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
        key = key * attention * query

        return query, key

#ADR
class ADR(torch.nn.Module):
    def __init__(self, in_plane1):
        super(ADR, self).__init__()

        self.bys = BYS(in_plane1)

        self.gcn = GloRe_Unit_2D(in_plane1, in_plane1, True)
        self.con_v1 = BasicConv2d(in_plane1, in_plane1 // 4, kernel_size=3, padding=1, dilation=1)
        self.con_v2 = BasicConv2d(in_plane1, in_plane1 // 4, kernel_size=3, padding=3, dilation=3)
        self.con_v3 = BasicConv2d(in_plane1, in_plane1 // 4, kernel_size=3, padding=5, dilation=5)
        self.con_v4 = BasicConv2d(in_plane1, in_plane1 // 4, kernel_size=3, padding=7, dilation=7)

    def forward(self, t):

        out_t = self.bys(t)

        t = self.gcn(out_t)
        value1 = self.con_v1(t)
        value2 = self.con_v2(t)
        value3 = self.con_v3(t)
        value4 = self.con_v4(t)
        out_t = torch.cat((value1, value2, value3, value4), dim=1)

        attention_w = torch.sigmoid(out_t)
        value = attention_w / torch.sum(attention_w, dim=(2, 3), keepdim=True)

        return value, out_t

#MWI
class MWI(nn.Module):
    def __init__(self, in_chanel):
        super(MWI, self).__init__()
        self.in_channel = in_chanel
        self.linear = nn.Linear(in_chanel, in_chanel, bias=False)
    def forward(self, rgb, qk, v, t):

        qkv = torch.sigmoid(qk * v)

        n = rgb.size()[0]
        c = rgb.size()[1]
        h = rgb.size()[2]
        w = rgb.size()[3]

        rgb_c_hw = rgb.view(n, c, -1)
        t_c_hw = t.view(n, c, -1)

        rgb_hw_c = self.linear(torch.transpose(rgb_c_hw, 1, 2).contiguous())
        t_hw_c = self.linear(torch.transpose(t_c_hw, 1, 2).contiguous())
        rgbt_hw_c = rgb_hw_c * t_hw_c
        rgbt_c_hw = torch.transpose(rgbt_hw_c, 1, 2).contiguous()
        rgbt_c_h_w = rgbt_c_hw.view(n, c, h, w) * qkv + qkv

        rgbt_hw_c = F.softmax(rgbt_c_hw, dim=2)
        rgbt_hw_c = F.softmax(rgbt_hw_c, dim=1)

        r_rgbt_hw_c = torch.transpose(rgbt_hw_c * rgb_c_hw, 1, 2).contiguous().view(n, c, h, w) + rgbt_c_h_w
        t_rgbt_hw_c = torch.sigmoid(torch.transpose(rgbt_hw_c * t_c_hw, 1, 2).contiguous()).view(n, c, h, w) * r_rgbt_hw_c

        return t_rgbt_hw_c

#Decoder+top-down relation fusion module
class Decoder(nn.Module):
    def __init__(self, in1, in2, in3, in4):
        super(Decoder, self).__init__()
        self.bcon4 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4, in4, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3 = BasicConv2d(in3, in4, kernel_size=3, stride=1, padding=1)
        self.bcon2 = BasicConv2d(in2, in3, kernel_size=3, stride=1, padding=1)
        self.bcon1 = BasicConv2d(in_planes=in1, out_planes=in2, kernel_size=1, stride=1, padding=0)

        self.bcon4_3 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4 * 2, in3, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3_2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4, in2, kernel_size=3, stride=1, padding=1)
        )
        self.bcon2_1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3, in1, kernel_size=3, stride=1, padding=1)
        )

        self.conv_d1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3, in2, kernel_size=3, stride=1, padding=1)
        )
        self.conv_d2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3, in1, kernel_size=3, stride=1, padding=1)
        )
        self.conv_d3 = BasicConv2d(in2, in1, kernel_size=3, stride=1, padding=1)

    def forward(self, f):
        f[3] = self.bcon4(f[3])
        f[2] = self.bcon3(f[2])
        f[1] = self.bcon2(f[1])
        f[0] = self.bcon1(f[0])

        d43 = self.bcon4_3(torch.cat((f[3], f[2]), 1))
        d32 = self.bcon3_2(torch.cat((d43, f[1]), 1))
        d21 = self.bcon2_1(torch.cat((d32, f[0]), 1))
        out1 = d21

        d43 = self.conv_d1(d43)
        d32 = torch.cat((d43, d32), dim=1)
        d32 = self.conv_d2(d32)
        d21 = torch.cat((d32, d21), dim=1)
        d21 = self.conv_d3(d21)
        out2 = d43
        out3 = d32

        return d21, out1, out2, out3


class Net_T(nn.Module):
    def __init__(self, ):
        super(Net_T, self).__init__()
        self.backbone = shunted_b()
        load_state_dict = torch.load(
            '/media/hjk/shuju/THNet_Distillation/THNet_Distillation/Net_pth/ckpt_B.pth')
        self.backbone.load_state_dict(load_state_dict)
        self.conv1_3 = BasicConv2d(1, 3, kernel_size=3, stride=1, padding=1)

        self.bqk1 = PGI(64)
        self.bqk2 = PGI(128)
        self.bqk3 = PGI(256)
        self.bqk4 = PGI(512)

        self.bv1 = ADR(64)
        self.bv2 = ADR(128)
        self.bv3 = ADR(256)
        self.bv4 = ADR(512)

        self.afm1 = MWI(64)
        self.afm2 = MWI(128)
        self.afm3 = MWI(256)
        self.afm4 = MWI(512)

        self.decoder = Decoder(64, 128, 256, 512)

        self.last1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(64, 1, 3, 1, 1)
        )
        self.last2 = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True),
            BasicConv2d(128, 1, 3, 1, 1)
        )

    def forward(self, x, y):
        merges = []
        rgb = self.backbone(x)
        t = self.backbone(self.conv1_3(y))
        # t = self.backbone(y)

        q1, k1 = self.bqk1(rgb[0])
        q2, k2 = self.bqk2(rgb[1])
        q3, k3 = self.bqk3(rgb[2])
        q4, k4 = self.bqk4(rgb[3])


        v1, t1 = self.bv1(t[0])
        v2, t2 = self.bv2(t[1])
        v3, t3 = self.bv3(t[2])
        v4, t4 = self.bv4(t[3])

        rgbt1 = self.afm1(q1, k1, v1, t1)
        rgbt2 = self.afm2(q2, k2, v2, t2)
        rgbt3 = self.afm3(q3, k3, v3, t3)
        rgbt4 = self.afm4(q4, k4, v4, t4)

        merges.append(rgbt1)
        merges.append(rgbt2)
        merges.append(rgbt3)
        merges.append(rgbt4)

        out, out1, out2, out3 = self.decoder(merges)

        return self.last1(out), self.last1(out1), self.last1(out3), self.last2(out2), rgbt1, rgbt2, rgbt3, rgbt4


if __name__ == '__main__':
    model = Net_T().cuda()
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
