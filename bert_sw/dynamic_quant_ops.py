import numpy as np
import torch


def tensor_quant_scale(arr, scale=None, bits=8):
    if scale is None:
#        arr_min = np.percentile(arr.detach().numpy(), .1)
#        arr_max = np.percentile(arr.detach().numpy(), 99.9)
#        scale = max(abs(arr_min), abs(arr_max)) / (2**(bits-1)-1)

        '''
        This is the quantiation method I picked up from pytorch/pytorch/aten/src/ATen/native/quantized/cpu/quant_utils.h
        '''
        arr_min = arr.min().item()
        arr_max = arr.max().item()
        qmax = 2**(bits-1) - 1
        qmin = -2**(bits-1)

        symmetric_qmin = -((qmax - qmin) / 2 + 1)
        symmetric_qmax = (qmax - qmin) / 2
        scale = max(abs(arr_min / symmetric_qmin), abs(arr_max / symmetric_qmax))

    
    arr_q = torch.round((torch.clip((arr / scale), min=-2**(bits-1)-1, max=2**(bits-1)-1)))
    return arr_q, scale


def tensor_quant_matmul(t1, t2):
    '''
    Quantizes two Tensors, performs INT8 matmul with INT32 accumulation, and returns a dequantized tensor.
    '''
    t1_q, t1_scale = tensor_quant_scale(t1)
    t2_q, t2_scale = tensor_quant_scale(t2)
    
    t3_q = torch.matmul(t1_q, t2_q)
    t3 = t3_q * t1_scale * t2_scale
    
    return t3.float()


def tensor_quant_linear(layer, act):
    act_q, act_scale = tensor_quant_scale(act)
    weight_q, weight_scale = tensor_quant_scale(layer.weight.T)
    
    acc_scale = act_scale * weight_scale
    bias_q, _ = tensor_quant_scale(layer.bias, scale=acc_scale, bits=32)
    
    ret_q = tensor_quant_matmul(act_q, weight_q) + bias_q
    
    ret = ret_q * acc_scale
    return ret.float()


def quant_scale(arr, scale=None, dtype=np.int8):
    if dtype == np.int8:
        bits = 8
    elif dtype == np.int32:
        bits = 32
    else:
        assert False, "Unsupported dtype for quantization"
        
    if scale is None:
        arr_min = np.percentile(arr, .1)
        arr_max = np.percentile(arr, 99.9)
        scale = max(abs(arr_min), abs(arr_max)) / (2**(bits-1)-1)
    
    arr_q = np.rint((arr / scale).clip(-2**(bits-1)-1, 2**(bits-1)-1)).astype(np.int32)
    return arr_q, scale


def quant_matmul(t1, t2):
    '''
    Quantizes two Tensors, performs INT8 matmul with INT32 accumulation, and returns a dequantized tensor.
    '''
    if not isinstance(t1, np.ndarray):
        t1 = t1.numpy()
    if not isinstance(t2, np.ndarray):
        t2 = t2.numpy()
        
    t1_q, t1_scale = quant_scale(t1)
    t2_q, t2_scale = quant_scale(t2)
    
    t3_q = np.matmul(t1_q, t2_q)
    
    t3 = t3_q * t1_scale * t2_scale
    
    return torch.from_numpy(t3).float()

def quant_linear(layer, act):
    '''
    
    '''
    act_q, act_scale = quant_scale(act.numpy())
    weight_q, weight_scale = quant_scale(layer.weight.T.detach().numpy())
    
    acc_scale = act_scale * weight_scale
    bias_q, _ = quant_scale(layer.bias.detach().numpy(), scale=acc_scale, dtype=np.int32)
    
    ret_q = quant_matmul(act_q, weight_q).numpy() + bias_q
    
    ret = ret_q * acc_scale
    return torch.from_numpy(ret).float()
    
    
    
