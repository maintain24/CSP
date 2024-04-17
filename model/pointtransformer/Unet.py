import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pointops.functions import pointops


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q = self.linear_q(x)
        x_k = self.linear_k(x)
        x_v = self.linear_v(x)
        # Simulate query and group operation
        x_k = torch.cat((p.unsqueeze(1).expand(-1, self.nsample, -1), x_k.unsqueeze(1)), dim=2)
        x_v = x_v.unsqueeze(1).expand(-1, self.nsample, -1)
        p_r, x_k = x_k[:, :, :3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r).sum(dim=2)  # Simplified version for demonstration
        w = x_k - x_q.unsqueeze(1) + p_r
        w = self.linear_w(w)
        w = self.softmax(w)
        x = (x_v * w).sum(dim=1)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.nsample = nsample
        self.stride = stride
        self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        if self.stride != 1:
            # Simplified sub-sampling for demonstration
            indices = torch.arange(0, x.size(0), self.stride, device=x.device)
            p, x = p[indices], x[indices]
        x = self.relu(self.bn(self.linear(x)))
        return p, x, o


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        self.linear = nn.Linear(in_planes, out_planes if out_planes is not None else in_planes)
        self.bn = nn.BatchNorm1d(out_planes if out_planes is not None else in_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo1, pxo2):
        p1, x1, o1 = pxo1
        p2, x2, o2 = pxo2
        x = self.relu(self.bn(self.linear(torch.cat([x1, x2], dim=1))))
        return p2, x, o2


class PointTransformerUNet(nn.Module):
    def __init__(self, c, k, num_classes=None):
        super(PointTransformerUNet, self).__init__()
        self.in_channels = c
        out_channels = c  # 在UNet中，通常最后一个解码器的输出与输入特征维度相同
        num_classes = k if num_classes is None else num_classes  # 如果没有单独指定num_classes，就使用k

        # 设置编码器
        self.enc1 = TransitionDown(c, 64)
        self.enc2 = TransitionDown(64, 128)
        self.enc3 = TransitionDown(128, 256)
        self.enc4 = TransitionDown(256, 512)

        # 中间的PointTransformerLayer
        self.middle = PointTransformerLayer(512, 512)

        # 设置解码器
        self.dec4 = TransitionUp(512 + 512, 256)
        self.dec3 = TransitionUp(256 + 256, 128)
        self.dec2 = TransitionUp(128 + 128, 64)
        self.dec1 = TransitionUp(64 + 64, out_channels)

        # 分类器，对每个点输出类别概率
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, p, x, o):
        # 编码路径
        pxo1 = self.enc1((p, x, o))
        pxo2 = self.enc2(pxo1)
        pxo3 = self.enc3(pxo2)
        pxo4 = self.enc4(pxo3)

        # 中间层
        pxo_middle = self.middle(pxo4)

        # 解码路径，使用跳跃连接
        d4 = self.dec4(pxo_middle, pxo4)
        d3 = self.dec3(d4, pxo3)
        d2 = self.dec2(d3, pxo2)
        d1 = self.dec1(d2, pxo1)

        # 输出类别
        logits = self.classifier(d1[1])
        return logits


# 创建模型实例
def unet_pointtransformer_seg(**kwargs):
    model = PointTransformerUNet(c=6, k=13)
    return model
