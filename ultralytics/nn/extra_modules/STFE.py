import torch
import torch.nn as nn
from ..modules.conv import Conv, autopad

def Upsample(x, size, align_corners=False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.avg_pool(x).view(b, c)  # squeeze操作
        x = self.fc(x).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x  # 注意力作用每一个通道上

class GLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.se = SE_Block(hidden_features)

    def forward(self, x):
        x_proj = self.fc1(x)
        x1, x2 = x_proj.chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x2 = self.se(x2)  # 通道注意力作用
        x = x1 * x2
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class STFE(nn.Module):
    def __init__(self, inc, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = Conv(inc[1], input_dim // 2, 1)
        self.d_in2 = Conv(inc[0], input_dim // 2, 1)

        self.conv = Conv(input_dim, input_dim, 3)

        # 直接对原始 H_feature 和 L_feature 进行门控
        self.glu_l = GLU(inc[1], out_features=input_dim // 2)
        self.glu_h = GLU(inc[0], out_features=input_dim // 2)

    def forward(self, x):
        H_feature, L_feature = x

        g_L_feature = torch.sigmoid(self.glu_l(L_feature))
        g_H_feature = torch.sigmoid(self.glu_h(H_feature))

        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature,
                                                                                       size=L_feature.size()[2:],
                                                                                       align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature,
                                                                                       size=H_feature.size()[2:],
                                                                                       align_corners=False)

        H_feature = Upsample(H_feature, size=L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out