from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 12
batch_size = 100000
n_gen = 100

# minimum and maximum which should be approximated.
x_min, x_max = -5, 5
name = 'sigmoid'


# change this to the function you want to approximate
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


mean = 0
stddev = 2
# the importance allows the user to add weights to the loss function.
# this way, certain regions can be made more important.
imp = 1


def inter(x):
    ret = np.interp(x,
                    xp=[x_min, mean - 3 * stddev, mean - 2 * stddev, mean, mean + 2 * stddev, mean + 3 * stddev, x_max],
                    fp=[0.5, 1, 10, 11, 10, 1, 0.5])
    return ret


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_scaled):
        ctx.save_for_backward(v_scaled)
        z_ = torch.where(v_scaled > 0, torch.ones_like(v_scaled), torch.zeros_like(v_scaled))
        return z_

    @staticmethod
    def backward(ctx, grad_output):
        v_scaled, = ctx.saved_tensors
        dz_dv_scaled = torch.clamp(1 - torch.abs(v_scaled), min=0)
        grad_input = grad_output * dz_dv_scaled
        return grad_input


spike_function = SpikeFunction.apply


class FSCoding(nn.Module):
    def __init__(self, K):
        super(FSCoding, self).__init__()
        self.K = K
        self.h = nn.Parameter(torch.rand(K).uniform_(-1, 0))
        self.d = nn.Parameter(torch.rand(K).uniform_(-0.5, 1))
        self.T = nn.Parameter(torch.rand(K).uniform_(-1, 1))

    def forward(self, x):
        v = x.clone()
        z = torch.zeros_like(x, device=device)
        out = torch.zeros_like(x, device=device)
        v_reg, z_reg = 0., 0.
        for t in range(self.K):
            v = v - z * self.h[t]  # update membrane potential
            v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)
            z = spike_function(v_scaled)  # spike function
            v_reg += torch.square(
                torch.mean(torch.maximum(torch.abs(v_scaled) - 1, torch.zeros_like(v_scaled))))  # regularization
            z_reg += torch.mean(z)
            out += z * self.d[t]  # compute output
        return out, v_reg, z_reg


model = FSCoding(K)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.1)
x = np.linspace(x_min, x_max, batch_size)
y = sigmoid(x)
x = torch.from_numpy(x).to(device)
y = torch.from_numpy(y).to(device)

# 创建一个数据集
dataset = TensorDataset(x, y)

# 定义一个数据加载器
dataloader = DataLoader(dataset, batch_size=500, shuffle=True)

best = 100000
current_best = 100000

for epoch in range(100000):
    train_epoch_loss = 0
    for i, (x_batch, y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        y_approx, v_reg, z_reg = model(x_batch)
        # loss = torch.mean(imp * torch.pow(torch.abs(y_batch - y_approx), 2.)) + \
        #        torch.rand(1, device=device).uniform_(0.08, 0.12) * v_reg + 0.00 * z_reg
        loss = criterion(y_approx, y_batch)
        loss.backward()
        optimizer.step()
        l = loss.item()
        # 累加每代中所有步数的loss
        train_epoch_loss += loss.item()
        if l < current_best:
            print(f"K: {K} i: {i} Time: {datetime.now()} Loss: {l} (v:{v_reg},z:{z_reg} )")
            current_best = l
    if train_epoch_loss < best:
        print("current_best:", current_best)
        print(f"K: {K} Epoch: {epoch} Time: {datetime.now()} Loss: {train_epoch_loss} (v:{v_reg},z:{z_reg} )")
        print(
            f"{name}_h = {np.array2string(model.h.detach().cpu().numpy(), separator=',')}\n{name}_d = {np.array2string(model.d.detach().cpu().numpy(), separator=',')}\n"
            f"{name}_T = {np.array2string(model.T.detach().cpu().numpy(), separator=',')}\n\n")
        best = train_epoch_loss
