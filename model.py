import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class LayerNorm2d(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.shape
        x_ = x.view(b, c, -1).transpose(1, 2)
        mean = x_.mean(dim=-1, keepdim=True)
        var = x_.var(dim=-1, keepdim=True, unbiased=False)
        x_ = (x_ - mean) / torch.sqrt(var + self.eps)
        x_ = x_ * self.weight + self.bias
        x_ = x_.transpose(1, 2).view(b, c, h, w)
        return x_


class ResBlock(nn.Module):

    def __init__(self, dim, res_scale=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))
        return x + self.res_scale * out


class CAFFBlock1(nn.Module):

    def __init__(self, dim, res_scale=1.0):
        super().__init__()
        self.res1 = ResBlock(dim, res_scale=res_scale)
        self.res2 = ResBlock(dim, res_scale=res_scale)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return x

class CAFFBlock2(nn.Module):

    def __init__(self, dim, res_scale=1.0):
        super().__init__()
        self.down = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, stride=2, bias=True)
        self.dw5 = nn.Conv2d(
            dim * 2, dim * 2,
            kernel_size=5, padding=2,
            groups=dim * 2, bias=True
        )
        self.pw1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, padding=0, bias=True)
        self.dilated3 = nn.Conv2d(
            dim * 2, dim * 2,
            kernel_size=3, padding=2, dilation=2,
            bias=True
        )
        self.pw2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, padding=0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.res = ResBlock(dim * 2, res_scale=res_scale)
        self.refine = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        base = x
        x_delta = self.act(self.dw5(base))
        x_delta = self.act(self.pw1(x_delta))
        x_delta = self.act(self.dilated3(x_delta))
        x_delta = self.act(self.pw2(x_delta))
        x = base + x_delta
        x = self.res(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.refine(x)
        return x


class ResidualAdaptiveFusion(nn.Module):

    def __init__(self, dim, reduction=4, fuse_scale=0.1):
        super().__init__()
        hidden_dim = max(dim // reduction, 4)
        self.base_fuse = nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.proj2 = nn.Conv2d(dim * 2, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.delta_gen = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=True)
        )
        nn.init.zeros_(self.delta_gen[-1].weight)
        nn.init.zeros_(self.delta_gen[-1].bias)
        self.fuse_scale = fuse_scale

    def forward(self, x1, x2):
        base = self.base_fuse(torch.cat((x1, x2), dim=1))
        x2_proj = self.proj2(x2)
        g1 = F.adaptive_avg_pool2d(x1, output_size=1)  
        g2 = F.adaptive_avg_pool2d(x2_proj, output_size=1)  
        gate = self.gate(torch.cat([g1, g2], dim=1))        
        delta_feat = torch.cat([base, x1, x2_proj], dim=1)
        delta = self.delta_gen(delta_feat)
        delta = gate * delta
        out = base + self.fuse_scale * delta
        return out


class CAFFBlock(nn.Module):

    def __init__(self, dim, res_scale=1.0, fuse_scale=0.1):
        super().__init__()
        self.tra1 = CAFFBlock1(dim, res_scale=res_scale)
        self.tra2 = CAFFBlock2(dim, res_scale=res_scale)
        self.fuse = ResidualAdaptiveFusion(dim=dim, reduction=4, fuse_scale=fuse_scale)

    def forward(self, x):
        identity = x
        x1 = self.tra1(x)
        x2 = self.tra2(x)
        out = self.fuse(x1, x2)
        return identity + out


class LCRA(nn.Module):

    def __init__(self, dim, attn_scale=0.05):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=True)
        )
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.attn_scale = attn_scale

    def forward(self, x):
        identity = x
        y = self.norm(x)
        a = self.a(y)
        v = self.v(y)
        y = a * v
        y = self.proj(y)
        return identity + self.attn_scale * y


class FRGUBlock(nn.Module):

    def __init__(self, dim, init_step=0.1, use_residual_refine=True):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)
        self.use_residual_refine = use_residual_refine
        if use_residual_refine:
            self.res1 = ResBlock(dim, res_scale=1.0)
        self.step = nn.Parameter(torch.tensor(init_step, dtype=torch.float32))

    def forward(self, x, y, Phi, PhiT):
        x_pixel = self.conv1(x)
        Phix = F.conv2d(x_pixel, Phi, padding=0, stride=32, bias=None)
        delta = y - Phix
        x_pixel = nn.PixelShuffle(32)(F.conv2d(delta, PhiT, padding=0, bias=None))
        x_delta = self.conv2(x_pixel)
        if self.use_residual_refine:
            x_delta = self.res1(x_delta)
        return x + self.step * x_delta


class CRAFTNet(nn.Module):

    def __init__(
        self,
        sensing_rate,
        LayerNo,
        base=16,
        phi_norm=False,
    ):
        super().__init__()
        self.measurement = int(sensing_rate * 1024)
        self.base = base
        self.LayerNo = LayerNo
        self.phi_norm = phi_norm
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.head = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.tail = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)
        grad_layers = []
        tra_layers = []
        attn_layers = []
        for _ in range(LayerNo):
            grad_layers.append(
                FRGUBlock(
                    dim=self.base,
                    use_residual_refine=True
                )
            )
            tra_layers.append(
                CAFFBlock(
                    dim=self.base,
                )
            )
            attn_layers.append(
                LCRA(
                    dim=self.base,
                )
            )

        self.grad_blocks = nn.ModuleList(grad_layers)
        self.tra_blocks = nn.ModuleList(tra_layers)
        self.attn_blocks = nn.ModuleList(attn_layers)

    def _get_phi(self):
        phi = self.Phi
        if self.phi_norm:
            phi = phi / (torch.norm(phi, dim=1, keepdim=True) + 1e-6)
        return phi

    def forward(self, x):
        phi = self._get_phi()
        Phi = phi.contiguous().view(self.measurement, 1, 32, 32)
        PhiT = phi.t().contiguous().view(1024, self.measurement, 1, 1)
        y = F.conv2d(x, Phi, padding=0, stride=32, bias=None)
        x = F.conv2d(y, PhiT, padding=0, bias=None)
        x = nn.PixelShuffle(32)(x)
        x = self.head(x)
        for i in range(self.LayerNo):
            x = self.grad_blocks[i](x, y, Phi, PhiT)
            x = self.tra_blocks[i](x)
            x = self.attn_blocks[i](x)
        x = self.tail(x)
        phi_cons = torch.mm(phi, phi.t())
        return x, phi_cons
