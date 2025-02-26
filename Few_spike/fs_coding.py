import torch

from Few_spike.fs_weight import *

n_neurons = 0
spike_num = 0
cnt = 0


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_scaled):
        ctx.save_for_backward(v_scaled)
        # 由wht修改成>=0
        z_ = torch.where(v_scaled >= 0, torch.ones_like(v_scaled), torch.zeros_like(v_scaled))
        return z_

    @staticmethod
    def backward(ctx, grad_output):
        v_scaled, = ctx.saved_tensors
        # 伪三角导数
        dz_dv_scaled = torch.clamp(1 - torch.abs(v_scaled), min=0)
        grad_input = grad_output * dz_dv_scaled
        return grad_input


spike_function = SpikeFunction.apply


def fs(x, h, d, T, K, return_reg=False, print_n_neurons=True):
    if print_n_neurons:
        global n_neurons
        # 模型中只有四个激活，一个sigmoid，三个relu
        global cnt
        if cnt < 3:
            n_neurons += np.prod(x.shape[1:])
            cnt += 1
        print(f'Number of neurons: {n_neurons}')
    v = x.clone()
    z = torch.zeros_like(x)
    out = torch.zeros_like(x)
    v_reg, z_reg, t = 0., 0., 0
    global spike_num
    while t < K:
        v_scaled = (v - T[t]) / (torch.abs(v) + 1)  # 控制缩放和调整数据的范围
        z = spike_function(v_scaled)
        v_reg += torch.square(torch.mean(torch.maximum(torch.abs(v_scaled) - 1, torch.zeros_like(v_scaled))))
        z_reg += torch.mean(z)
        out += z * d[t]
        v = v - z * h[t]
        t = t + 1

    if return_reg:
        return out, v_reg, z_reg
    else:
        return out


def fs_sigmoid(x, return_reg=False):
    return fs(x, torch.Tensor(sigmoid_h), torch.Tensor(sigmoid_d), torch.Tensor(sigmoid_T), K=len(sigmoid_h),
              return_reg=return_reg)


def replace_relu_with_fs():
    '''
    Call this function to replace the ReLU functions with an FS-neuron which approximates a ReLU function.
    '''

    def custom_activation(x, inplace=False):
        return fs_relu(x, fast=True)

    torch.nn.functional.relu = custom_activation


def replace_sigmoid_with_fs():
    '''
    Call this function to replace the Sigmoid functions with an FS-neuron which approximates a Sigmoid function.
    '''

    def custom_activation(x, inplace=False):
        return fs_sigmoid(x)

    torch.sigmoid = custom_activation


def fs_relu(x, n_neurons=10, v_max=9, return_reg=False, fast=False):
    '''
    Note: As the relu function is a special case, it is no necessary to use the fs() function.
    It is computationally cheaper to simply discretize the input and clip to the
    minimum and maximum.
    '''
    if fast:
        x = torch.clamp(x, 0)
        x /= v_max

        x *= 2 ** n_neurons
        i_out = torch.floor(x)
        i_out /= 2 ** n_neurons
        i_out *= v_max
        i_out = torch.min(i_out, torch.tensor(v_max * (1 - 2 ** (-n_neurons)), dtype=i_out.dtype))
        if return_reg:
            return i_out, torch.tensor(1.)
        return i_out
    else:
        return fs(x, relu_h, relu_d, relu_T, K=len(relu_h), return_reg=return_reg)
