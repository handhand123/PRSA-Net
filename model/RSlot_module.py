import torch
import torch.nn as nn

class RSlot_module(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            pos,
            groups=32,
            t_scale=200,
            adjacency_snippets='4,8',
            atten=False
    ):
        super(RSlot_module, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.atten = atten
        self.hid = 128
        self.pos = torch.tensor(pos, dtype=torch.float32, requires_grad=False)

        self.pe = PE(in_channel, self.hid, out_channel, groups, self.pos)

        self.global_attn = global_conv_attn(in_channel, self.hid, out_channel, groups)

        self.PRSlot = PRSlot_Moule(in_channel, self.hid, out_channel, groups, adjacency_snippets, t_scale)

        self.relu = nn.ReLU(True)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x_g = self.global_attn(x)

        x_p = self.pe(x)

        x_s, conns = self.PRSlot(x)

        x_s = self.bn(x_g + x_p + x_s)

        return self.relu(x_s), conns

def _get_Convs(in_channel, hid_channel, out_channel, groups=32):
    return nn.Sequential(
        nn.Conv1d(in_channel, hid_channel, 1, 1),
        nn.Conv1d(hid_channel, hid_channel, kernel_size=3, groups=groups, padding=1),
        nn.ReLU(True),
        nn.Conv1d(hid_channel, out_channel, kernel_size=1)
    )

def slot_mask(adjacency_snippet, scale):
    mask = torch.zeros(scale, scale)
    for i in range(scale):
        for j in range(scale):
            if j >= i - adjacency_snippet and j <= i + adjacency_snippet:
                mask[i][j] = 1
    return mask

class PRSlot_block(nn.Module):
    def __init__(self, in_channel, out_channel, adjacency_snippets, t_scale):
        super(PRSlot_block, self).__init__()
        kernal_size = (2 * adjacency_snippets + 1, 1)
        stride = (1, 1)
        padding = (adjacency_snippets, 0)
        self.proj = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.region_key_conv = nn.Conv2d(out_channel, 128, kernel_size=kernal_size, stride=stride, padding=padding)
        self.aggregate_conv = nn.Conv2d(128, 1, kernel_size=kernal_size, stride=stride, padding=padding)
        self.softmax = nn.Softmax(-2)
        self.slot_mask = slot_mask(adjacency_snippets, t_scale)
        self.slot_mask = torch.tensor(self.slot_mask, dtype=torch.float32, requires_grad=False)

    def forward(self, x):
        x = self.proj(x)

        x = self.region_key_conv(x)
        if self.slot_mask is not None:
            self.slot_mask = self.slot_mask.cuda(x.get_device())
            x = torch.mul(x, self.slot_mask)
        x = self.aggregate_conv(x).squeeze(1).contiguous()
        x = self.softmax(x)
        return x

class global_conv_attn(nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel, groups):
        super(global_conv_attn, self).__init__()
        self.proj_convs_v = _get_Convs(in_channel, hid_channel, out_channel, groups)
        self.proj_convs_k = nn.Conv1d(in_channel, hid_channel, 1)
        self.proj_convs_q = nn.Conv1d(in_channel, hid_channel, 1)
        self.softmax = nn.Softmax(-2)


    def forward(self, x):
        k = self.proj_convs_k(x).permute(0, 2, 1).contiguous()
        q = self.proj_convs_q(x)
        sim = self.softmax(torch.matmul(k, q))
        v = self.proj_convs_v(x)
        
        return torch.matmul(v, sim)

class PE(nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel, groups, pos):
        super(PE, self).__init__()
        self.positional_convs = _get_Convs(in_channel, hid_channel, out_channel, groups)
        self.pos = pos

    def forward(self, x):
        pos = self.pos.cuda(x.get_device())
        x = self.positional_convs(x)

        return torch.matmul(x, pos)


class PRSlot_Moule(nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel, groups, adjacency_snippets, t_scale):
        super(PRSlot_Moule, self).__init__()
        self.semantic_convs = _get_Convs(in_channel, hid_channel, out_channel, groups)
        self.PyramidRegionSlots = nn.ModuleList()
        self.adjacency_snippets = adjacency_snippets
        for i in range(len(adjacency_snippets)):
            self.PyramidRegionSlots.append(PRSlot_block(in_channel, hid_channel, adjacency_snippets[i], t_scale))

    def forward(self, x):
        N, C, T = x.shape

        PRSlot_matrix = []
        x_adj = x.unsqueeze(3).repeat(1, 1, 1, T)

        for i in range(len(self.adjacency_snippets)):
            PRSlot_matrix.append(self.PyramidRegionSlots[i](x_adj))
        PRSlot_matrix = torch.stack(PRSlot_matrix, dim=0).sum(dim=0)

        x = self.semantic_convs(x)
        x = torch.matmul(x, PRSlot_matrix)

        return x, PRSlot_matrix