import torch
from .quant_ops import tensor_quant_scale

def linear_kernel(act, weight_t, bias):
    '''
    Linear Layer generalizes to matmul, you just initialize accumulators to zero.

    act:        int8 quantized activation
    weight_t:   int8 quantized weights (transposed)
    bias:       int32 quantized bias
    acc_scale:  scaling factor used for accumulation/bias (act_scale*weight_scale)
    out_scale:  scaling factor used on output

    returns:    int32 quantized output (because it's more general than 8-bit)
    '''
    
    weight_t = (weight_t.type(torch.int8)).type(torch.int32)
    bias = bias.type(torch.int32)
    act = (act.type(torch.int8)).type(torch.int32)

    acc = torch.matmul(act, weight_t) + bias

    return acc



