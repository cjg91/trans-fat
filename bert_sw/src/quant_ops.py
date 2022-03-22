import numpy as np
import torch

DOCSTRING = '''

In this file we write software implementations of the quantized operations of BERT. We then replace those operations in dynamic_quant_roberta.py and demonstrate minimal performance hit by evaluating in 3-quant_roberta.ipynb.

'''


def tensor_quant_scale(arr, scale=None, bits=8):
    '''
    arr:   Tensor
    scale: scaling factor
    bits:  bit width
    '''
    if scale is None:
        '''
        This is the quantization method I picked up from pytorch/pytorch/aten/src/ATen/native/quantized/cpu/quant_utils.h
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
    
    # INT8 territory
    
    # note: performing matmul on float tensors that contain integers in the INT8 range.
    # float matmul is optimized for CPU, int matmul is not.
    t3_q = torch.matmul(t1_q, t2_q)
    
    # INT32 territory
    # Make sure we stayed in INT32 range
    assert t3_q.max().item() < 2**31-1, t3_q.max().item()
    assert t3_q.min().item() > -2**31

    t3 = t3_q * t1_scale * t2_scale
    
    # Float territory
    
    return t3.float()


def tensor_quant_linear(layer, act):
    '''
    layer: linear layer (nn.Linear)
    act:   Tensor, input to linear layer
    
    returns: dequantized float tensor
    '''
    act_q, act_scale = tensor_quant_scale(act)
    weight_q, weight_scale = tensor_quant_scale(layer.weight.T)
    
    acc_scale = act_scale * weight_scale
    bias_q, _ = tensor_quant_scale(layer.bias, scale=acc_scale, bits=32)
    
    ret_q = torch.matmul(act_q, weight_q) + bias_q
    
    assert ret_q.max().item() < 2**31-1, ret_q.max().item()
    assert ret_q.min().item() > -2**31
    
    ret = ret_q * acc_scale
    return ret.float()


def tensor_quant_gelu(act):
    '''
    Copied this from I-BERT implementation. Yields >20% accuracy drop.
    '''
    k = 1.4142
    const = 14
    coeff = [-0.2888, -1.769, 1]
    coeff[2] /= coeff[0]
    
    def int_erf(x_int, scaling_factor):
        b_int = torch.round(torch.Tensor([coeff[1] / scaling_factor]))
        c_int = torch.round(torch.Tensor([coeff[2] / scaling_factor**2]))
        sign = torch.sign(x_int)

        abs_int = torch.min(torch.abs(x_int), -b_int)
        y_int = sign * ((abs_int + b_int) ** 2 + c_int)
        scaling_factor = scaling_factor**2 * coeff[0]

        # avoid overflow
        y_int = torch.round(y_int / 2**const)
        scaling_factor = scaling_factor * 2**const

        return y_int, scaling_factor
    
    x_int, scaling_factor = tensor_quant_scale(act, bits=32)
    sigmoid_int, sigmoid_scaling_factor = int_erf(x_int, scaling_factor / k)
    
    shift_int = torch.round(torch.Tensor([1.0 / sigmoid_scaling_factor]))

    x_int = x_int * (sigmoid_int + shift_int)
    scaling_factor = scaling_factor * sigmoid_scaling_factor / 2

    return x_int * scaling_factor

#     return torch.nn.functional.gelu(act)

    

## TODO: Implement quantized operations

def tensor_quant_softmax(act):
    '''
    TODO: quantize input and implement integer softmax
    '''
#     return torch.nn.functional.softmax(act, dim=-1)
    output_bit = 8
    max_bit = 32
    x0 = -0.6931  # -ln2
    const = 30  # dummy integer constant
    coef = [0.35815147, 0.96963238, 1.0]  # ax**2 + bx + c
    coef[1] /= coef[0]
    coef[2] /= coef[0]

    def int_polynomial(x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(torch.Tensor([coef[1] / scaling_factor]))
            c_int = torch.floor(torch.Tensor([coef[2] / scaling_factor**2]))
        z = (x_int + b_int) * x_int + c_int
        scaling_factor = coef[0] * scaling_factor**2
        return z, scaling_factor
    
    def int_exp(x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(torch.Tensor([x0 / scaling_factor]))
        x_int = torch.max(x_int, const * x0_int)

        q = torch.floor(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(torch.floor(exp_int * 2 ** (const - q)), min=0)
        scaling_factor = exp_scaling_factor / 2**const
        return exp_int, scaling_factor

    
    x_int, scaling_factor = tensor_quant_scale(act, bits=32)

    x_int_max, _ = x_int.max(dim=-1, keepdim=True)
    x_int = x_int - x_int_max
    exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)

    # Avoid overflow
#     exp, exp_scaling_factor = self.act(exp_int, exp_scaling_factor)
#     exp_int = exp / exp_scaling_factor
    exp = exp_int * exp_scaling_factor
    exp_int, exp_scaling_factor = tensor_quant_scale(exp, bits=16)

    exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
    factor = torch.floor(2**max_bit / exp_int_sum)
    exp_int = torch.floor(exp_int * factor / 2 ** (max_bit - output_bit))
    scaling_factor = 1 / 2**output_bit
    return exp_int * scaling_factor


def tensor_quant_layernorm(layernorm, act):
    '''
    TODO: quantize input and implement integer layernorm
    layernorm: nn.LayerNorm module
    act:       Float Tensor
    '''

    '''Note: I was doing this wrong even without quantization, so fix that first and then quantize the operations.'''
    # return torch.nn.functional.layer_norm(act, layernorm.normalized_shape, layernorm.weight, layernorm.bias, layernorm.eps)

    act_int, scaling_factor = tensor_quant_scale(act, bits=16)
    weight_int, _ = tensor_quant_scale(layernorm.weight.data, scale=scaling_factor, bits=16)
    bias_int, _ = tensor_quant_scale(layernorm.bias.data, scale=scaling_factor, bits=16)
    
    n = torch.tensor(act.shape[2], dtype=torch.float)
    
    mean_int = torch.round(torch.mean(act_int, axis=2, keepdim=True))
    y_int = act_int - mean_int
    y_sq_int = torch.round(y_int**2 / n)
    var_int = torch.sum(y_sq_int, axis=2, keepdim=True) 
    
    assert var_int.max() < 2**31 + .01
    std_int = torch.round(torch.sqrt(var_int + int(layernorm.eps / scaling_factor)))

    ###############
    # Unquantized Operation
    y_int = y_int / std_int
    ###############
    
    y_int = y_int * weight_int + bias_int
    y = y_int * scaling_factor
    return y
    
    
    
    '''
    Version 2
    shift = 2**18
    
    x_int, scaling_factor = tensor_quant_scale(act, bits=16)
    weight_int, _ = tensor_quant_scale(layernorm.weight.data, bits=16, scale=scaling_factor)
    bias_int, _ = tensor_quant_scale(layernorm.bias.data, bits=16, scale=scaling_factor)
    
    mean_int = torch.round(torch.mean(x_int, axis=2, keepdim=True))
    y_int = x_int - mean_int
    y_int_shifted = y_int / shift
    y_sq_int = y_int_shifted**2
    var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
    
    assert var_int.max() < 2**31 + .01
    
    n = torch.tensor(act.shape[2], dtype=torch.float)
    dim_sqrt = torch.sqrt(n).to(act.device)
    
    std_int = torch.floor(torch.sqrt(var_int)* shift)
    y_int = torch.floor((y_int * dim_sqrt) / std_int)
    
    y_int = y_int * weight_int + bias_int
    out = y_int * scaling_factor
    return out
    '''

    
    
    
    


# Version 1
    # compute sqrt of the feature dimension if it is the first run

#     shift = 16 # 16, 17, and 18 are common shifts
#     max_bit = 32
    
#     def set_shift(y_int, shift):
#         with torch.no_grad():
#             y_sq_int = y_int**2
#             var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
#             shift_new = (torch.log2(torch.sqrt(var_int / 2**max_bit)).ceil()).max()
#             shift = max(shift, shift_new)
#         print(f"Dynamic shift adjustment: {int(shift)}")

#         return shift

    
#     def overflow_fallback(y_int, shift):
#         """
#         This fallback function is called when overflow is detected during training time, and adjusts the `self.shift`
#         to avoid overflow in the subsequent runs.
#         """
#         shift = set_shift(y_int, shift)  # adjusts `self.shift`
#         y_int_shifted = torch.floor(y_int / 2**shift)
#         y_sq_int = y_int_shifted**2
#         var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
#         return var_int, shift
    
#     ## Get dim_sqrt (can be computed only once)
#     n = torch.tensor(act.shape[2], dtype=torch.float)
#     dim_sqrt = torch.sqrt(n).to(act.device)

#     # Normalization: computes mean and variance(std)
#     x_int, scaling_factor = tensor_quant_scale(act, bits=32)
    
#     mean_int = torch.round(x_int.mean(axis=2, keepdim=True))
#     y_int = x_int - mean_int
#     y_int_shifted = torch.floor(y_int / 2**shift)
#     y_sq_int = y_int_shifted**2
#     var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
    
#     if var_int.max() >= 2**max_bit:
#         var_int, shift = overflow_fallback(y_int, shift)
        
#         assert var_int.max() < 2**max_bit + 0.1, (
#             "Error detected in overflow handling: "
#             "`var_int` exceeds `self.max_bit` (the maximum possible bit width)"
#         )

#     # To be replaced with integer-sqrt kernel that produces the same output
#     std_int = torch.floor(torch.sqrt(var_int)) * 2**shift
#     factor = torch.floor(2**31 / std_int)
#     y_int = torch.floor(y_int * factor / 2)
#     scaling_factor = dim_sqrt / 2**30

#     # scaling and shifting
# #     bias = layernorm.bias.data.detach() / (layernorm.weight.data.detach())
#     bias_int = torch.floor(layernorm.bias.data.detach() / scaling_factor)
#     bias_int = torch.floor(layernorm.bias.data.detach() / scaling_factor)

#     y_int = y_int + bias_int
#     scaling_factor = scaling_factor * layernorm.weight
#     x = y_int * scaling_factor

#     return x




def tensor_quant_addition(in1, in2):
    '''
    Do we need to implement this as well?
    '''
    pass



## Deprecated

# def quant_scale(arr, scale=None, dtype=np.int8):
#     if dtype == np.int8:
#         bits = 8
#     elif dtype == np.int32:
#         bits = 32
#     else:
#         assert False, "Unsupported dtype for quantization"
        
#     if scale is None:
#         arr_min = np.percentile(arr, .1)
#         arr_max = np.percentile(arr, 99.9)
#         scale = max(abs(arr_min), abs(arr_max)) / (2**(bits-1)-1)
    
#     arr_q = np.rint((arr / scale).clip(-2**(bits-1)-1, 2**(bits-1)-1)).astype(np.int32)
#     return arr_q, scale


# def quant_matmul(t1, t2):
#     '''
#     Quantizes two Tensors, performs INT8 matmul with INT32 accumulation, and returns a dequantized tensor.
#     '''
#     if not isinstance(t1, np.ndarray):
#         t1 = t1.numpy()
#     if not isinstance(t2, np.ndarray):
#         t2 = t2.numpy()
        
#     t1_q, t1_scale = quant_scale(t1)
#     t2_q, t2_scale = quant_scale(t2)
    
#     t3_q = np.matmul(t1_q, t2_q)
    
#     t3 = t3_q * t1_scale * t2_scale
    
#     return torch.from_numpy(t3).float()

# def quant_linear(layer, act):
#     '''
    
#     '''
#     act_q, act_scale = quant_scale(act.numpy())
#     weight_q, weight_scale = quant_scale(layer.weight.T.detach().numpy())
    
#     acc_scale = act_scale * weight_scale
#     bias_q, _ = quant_scale(layer.bias.detach().numpy(), scale=acc_scale, dtype=np.int32)
    
#     ret_q = quant_matmul(act_q, weight_q).numpy() + bias_q
    
#     ret = ret_q * acc_scale
#     return torch.from_numpy(ret).float()

    
    
    
