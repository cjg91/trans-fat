import torch

'''
These kernels differ from ops because they have well-defined interfaces and operate mainly on integer types.
These will be mapped, almost exactly, to HLS kernels.
'''

def linear_kernel(act: torch.CharTensor, weight_t: torch.CharTensor, bias: torch.IntTensor) -> torch.IntTensor:
    '''
    Linear

    act:        int8 quantized activation
    weight_t:   int8 quantized weights (transposed)
    bias:       int32 quantized bias

    returns:    int32 quantized output (because it's more general than 8-bit)
    '''
    
    weight_t = (weight_t.type(torch.int8)).type(torch.int32)
    bias = bias.type(torch.int32)
    act = (act.type(torch.int8)).type(torch.int32)

    acc = torch.matmul(act, weight_t) + bias

    return acc


def matmul_kernel(A: torch.CharTensor, B: torch.CharTensor) -> torch.IntTensor:
    '''
    Matmul

    A:          int8 quantized A
    B:          int8 quantized B

    returns:    int32 quantized output (because it's more general than 8-bit)
    '''
    
    # Make sure we're dealing in INTs here
    A = (A.type(torch.int8)).type(torch.int32)
    B = (B.type(torch.int8)).type(torch.int32)

    acc = torch.matmul(A, B)

    return acc


def requantize_kernel(A: torch.IntTensor, M_scale) -> torch.CharTensor:
    '''
    A:      int32 tensor
    scale:  fixed-point scaling factor to quantize to int8
    '''
    bits_out = 8

    A_int = A.float() * M_scale # FixedPoint Multiply
    A_int = torch.round((torch.clip((A_int), min=-2**(bits_out-1)-1, max=2**(bits_out-1)-1)))
    return A_int.type(torch.int8)



