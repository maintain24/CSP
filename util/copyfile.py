# -*- coding:utf-8 -*-
'''
尝试用gpt来改写部分函数
'''
class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, share_planes=8, nsample=16):
        super().__init__()

        mid_channels = out_channels // 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_channels, mid_channels)
        self.linear_k = nn.Linear(in_channels, mid_channels)
        self.linear_v = nn.Linear(in_channels, out_channels)

        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_channels)
        )

        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels // share_planes),
            nn.BatchNorm1d(mid_channels // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // share_planes, out_channels // share_planes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, xyz_feature):
        # Get input values
        xyz, feature = xyz_feature
        # Compute query, key and value features
        q = self.linear_q(feature)
        k = self.linear_k(feature)
        v = self.linear_v(feature)
        # Group query features by nearest neighbors
        k = query_and_group(xyz, xyz, k, self.nsample)
        # Group value features by nearest neighbors
        v = query_and_group(xyz, xyz, v, self.nsample)
        # Split position vectors and key features
        pos, k = k[:, :, :3], k[:, :, 3:]
        # Apply linear transformations to position vectors
        for i, layer in enumerate(self.linear_p):
            if i == 1:
                pos = layer(pos.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            else:
                pos = layer(pos)
        # Compute weight vectors
        w = k - q.unsqueeze(1) + pos.view(pos.shape[0], pos.shape[1], self.out_channels // self.in_channels,
                                          self.in_channels).sum(2)
        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        # Apply softmax to weight vectors
        w = self.softmax(w)
        # Reshape value features and compute output
        n, nsample, c = v.shape
        s = self.share_planes
        x = ((v + pos).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        self.out_planes = out_planes or in_planes
        self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, 2 * self.out_planes), nn.BatchNorm1d(2 * self.out_planes),
                                     nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(self.out_planes, self.out_planes), nn.ReLU(inplace=True))

    def forward(self, x1, x2=None):
        if x2 is None:
            b = x1.shape[0]
            c = self.out_planes
            o = torch.arange(b).repeat_interleave(c).view(b, c)
            x_tmp = []
            for i in range(b):
                x_b = x1[i].unsqueeze(0)
                x_b = torch.cat((x_b, self.linear2(x_b.mean(dim=0)).unsqueeze(0)), dim=1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x.view(-1, 2 * self.out_planes)).view(b, c, -1).transpose(1, 2).reshape(-1, c)
        else:
            x = self.linear1(x1) + F.interpolate(self.linear2(x2), scale_factor=2, mode='nearest')
        return x
