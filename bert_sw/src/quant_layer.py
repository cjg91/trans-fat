import torch
import torch.nn as nn
import math
from .quant_ops import (
    tensor_quant_matmul, tensor_quant_linear, tensor_quant_softmax,
    tensor_quant_layernorm, tensor_quant_gelu, tensor_quant_scale
)
from .quant_kernels import linear_kernel, matmul_kernel, requantize_kernel

def quantize_linear_params(layer, act_scale):
    weight, weight_scale = tensor_quant_scale(layer.weight, bits=8)
    acc_scale = weight_scale * act_scale
    bias, _ = tensor_quant_scale(layer.bias, scale=acc_scale, bits=32)

    return weight, bias, acc_scale


def stage1(act_int, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias, M_query, M_key, M_value):
    '''
    act_int:                                        int8 quantized activation (input)
    query_weight_t, key_weight_t, value_weight_t:   int8 quantized weights (already transposed) for linear transforms
    query_bias, key_bias, value_bias:               int32 quantized biases for linear transforms
    M_query, M_key, M_value:                        requantization values
    '''

    query_layer = linear_kernel(act_int, query_weight_t, query_bias) # <bs, seqlen, dmodel>
    key_layer = linear_kernel(act_int, key_weight_t, key_bias) # <bs, seqlen, dmodel>
    value_layer = linear_kernel(act_int, value_weight_t, value_bias) # <bs, seqlen, dmodel>

    query_layer = requantize_kernel(query_layer, M_query)
    key_layer = requantize_kernel(key_layer, M_key)
    value_layer = requantize_kernel(value_layer, M_value)

    return query_layer, key_layer, value_layer


def stage2(query, value, key, scores_scale, M_attention_probs, M_attention_out, dense_weight_t, dense_bias, dense_acc_scale, layernorm, skip_conn, M_stage2):
    '''
    query:  <bs, seqlen, dmodel> int8 quantized query
    value:  <bs, seqlen, dmodel> int8 quantized value
    key:    <bs, seqlen, dmodel> int8 quantized key
    '''

    bs = 1
    seqlen = 128
    num_heads = 12
    dhead = 64
    dmodel = 768
    new_shape = (bs, seqlen, num_heads, dhead)
    
    query = query.view(new_shape)
    value = value.view(new_shape)
    key = key.view(new_shape)
    
    query = query.permute(0,2,1,3) # <bs, num_head, seqlen, dhead>
    value = value.permute(0,2,1,3) # <bs, num_head, seqlen, dhead>
    key = key.permute(0,2,3,1)     # <bs, num_head, dhead, seqlen>
    
    attention_scores_int = matmul_kernel(query, key)

    # dequantize
    attention_scores = attention_scores_int.float() * scores_scale
    attention_scores /= math.sqrt(dhead)

    attention_probs = tensor_quant_softmax(attention_scores)

    # requantize
    attention_probs_int = requantize_kernel(attention_probs, M_attention_probs)

    attention_out = matmul_kernel(attention_probs_int, value)
    attention_out = requantize_kernel(attention_out, M_attention_out)

    attention_out = attention_out.permute(0,2,1,3).contiguous()
    attention_out = attention_out.view(bs, seqlen, dmodel)

    dense_out = linear_kernel(attention_out, dense_weight_t, dense_bias)

    # dequantize
    dense_out = dense_out.float() * dense_acc_scale
    
    dense_out = dense_out + skip_conn

    attention_out = tensor_quant_layernorm(layernorm, dense_out)
    
    # requantize
    attention_out = requantize_kernel(attention_out, M_stage2)

    return attention_out


def stage3(fc_in, dense_weight_t, dense_bias, dense_acc_scale, M_stage3):
    output = linear_kernel(fc_in, dense_weight_t, dense_bias)
    
    # dequantize 
    output = output.float() * dense_acc_scale
    output = tensor_quant_gelu(output)
    output = requantize_kernel(output, M_stage3)

    return output


def stage4(fc_in, dense_weight_t, dense_bias, dense_acc_scale, skip_conn, layernorm, M_stage4):
    fc_out = linear_kernel(fc_in, dense_weight_t, dense_bias)
    
    # dequantize
    fc_out = fc_out.float() * dense_acc_scale

    fc_out += skip_conn
    fc_out = tensor_quant_layernorm(layernorm, fc_out)

    # requantize
    fc_out = requantize_kernel(fc_out, M_stage4)
    
    return fc_out

def layer_kernel_gt(layer, hidden_states):
    '''
    Demonstrates pipeline stages in the encoder layer. Prepares inputs and scaling factors
    and uses stages to perform computation.
    '''

    bs, seqlen, dmodel = hidden_states.size()
    num_heads = layer.attention.self.num_attention_heads
    dhead = layer.attention.self.attention_head_size
    
    ###############
    # Stage 1
    ###############
    stage1_args = {}
    # Preparing the inputs (would not be done in hardware)
    act_int, act_scale = tensor_quant_scale(hidden_states)
    query_weight, query_bias, query_acc_scale = quantize_linear_params(layer.attention.self.query, act_scale)
    key_weight, key_bias, key_acc_scale = quantize_linear_params(layer.attention.self.key, act_scale)
    value_weight, value_bias, value_acc_scale = quantize_linear_params(layer.attention.self.value, act_scale)

    query_layer = linear_kernel(act_int, query_weight.T, query_bias) # <bs, seqlen, dmodel>
    key_layer = linear_kernel(act_int, key_weight.T, key_bias) # <bs, seqlen, dmodel>
    value_layer = linear_kernel(act_int, value_weight.T, value_bias) # <bs, seqlen, dmodel>

    _, query_out_scale = tensor_quant_scale(query_layer * query_acc_scale)
    _, key_out_scale = tensor_quant_scale(key_layer * key_acc_scale)
    _, value_out_scale = tensor_quant_scale(value_layer * value_acc_scale)

    M_query = query_acc_scale / query_out_scale
    M_key = key_acc_scale / key_out_scale
    M_value = value_acc_scale / value_out_scale

    stage1_args['act_int'] = act_int
    stage1_args['query_weight_t'] = query_weight.T
    stage1_args['query_bias'] = query_bias
    stage1_args['key_weight_t'] = key_weight.T
    stage1_args['key_bias'] = key_bias
    stage1_args['value_weight_t'] = value_weight.T
    stage1_args['value_bias'] = value_bias
    stage1_args['M_query'] = M_query
    stage1_args['M_key'] = M_key
    stage1_args['M_value'] = M_value

    ###############
    # Stage 2
    ###############
    stage2_args = {}
    # stage2_args['query'] = query_layer
    # stage2_args['key'] = key_layer
    # stage2_args['value'] = value_layer
    stage2_args['scores_scale'] = (query_out_scale * key_out_scale)
    stage2_args['layernorm'] = layer.attention.output.LayerNorm
    stage2_args['skip_conn'] = hidden_states
    
    new_shape = (bs, seqlen, num_heads, dhead)
    
    query_layer = query_layer.view(new_shape)
    value_layer = value_layer.view(new_shape)
    key_layer = key_layer.view(new_shape)
    
    query_layer = query_layer.permute(0,2,1,3) # <bs, num_head, seqlen, dhead>
    value_layer = value_layer.permute(0,2,1,3) # <bs, num_head, seqlen, dhead>
    key_layer = key_layer.permute(0,2,3,1)     # <bs, num_head, dhead, seqlen>
    
    attention_scores = matmul_kernel(query_layer, key_layer)
    
    # DEQUANT ZONE BEGIN ⬇️`
    attention_scores = attention_scores.float() * (query_out_scale * key_out_scale)
    attention_scores /= math.sqrt(dhead)
    
    attention_probs = tensor_quant_softmax(attention_scores)
    # DEQUANT ZONE END ^^^^^^^^^^
    # quantize
    attention_probs, attention_prob_scale = tensor_quant_scale(attention_probs, bits=8)
    attention_probs = attention_probs.type(torch.int8)
    stage2_args['M_attention_probs'] = (1/attention_prob_scale)

    attention_out = matmul_kernel(attention_probs, value_layer)

    # requantize
    _, attention_out_scale = tensor_quant_scale(attention_out*(attention_prob_scale * value_out_scale), bits=8)
    M_attention_out = attention_prob_scale * value_out_scale / attention_out_scale
    stage2_args['M_attention_out'] = M_attention_out
    attention_out = requantize_kernel(attention_out, M_attention_out)
    
    attention_out = attention_out.permute(0,2,1,3).contiguous()
    attention_out = attention_out.view(bs, seqlen, dmodel)
    
    dense_weight, dense_bias, dense_acc_scale = quantize_linear_params(layer.attention.output.dense, attention_out_scale)
    stage2_args['dense_weight_t'] = dense_weight.T
    stage2_args['dense_bias'] = dense_bias
    stage2_args['dense_acc_scale'] = dense_acc_scale
    dense_out = linear_kernel(attention_out, dense_weight.T, dense_bias)

    dense_out = dense_out.float() * dense_acc_scale
    
    # DEQUANT ZONE BEGIN ⬇️`

    attention_out = tensor_quant_layernorm(layer.attention.output.LayerNorm, dense_out + hidden_states)

    # DEQUANT ZONE END ^^^^^^^^^^

    ###############
    # Stage 3
    ###############
    stage3_args = {}

    attention_out_int, attention_out_scale = tensor_quant_scale(attention_out, bits=8)
    stage2_args['M_stage2'] = (1/attention_out_scale)
    attention_out_int = attention_out_int.type(torch.int8)
    dense_weight, dense_bias, dense_acc_scale = quantize_linear_params(layer.intermediate.dense, attention_out_scale)
    stage3_args['dense_weight_t'] = dense_weight.T
    stage3_args['dense_bias'] = dense_bias
    stage3_args['dense_acc_scale'] = dense_acc_scale

    output = linear_kernel(attention_out_int, dense_weight.T, dense_bias)
    
    # DEQUANT ZONE BEGIN ⬇️`
    output = output.float() * dense_acc_scale
    output = tensor_quant_gelu(output)

    # DEQUANT ZONE END ^^^^^^^^^^
    
    ###############
    # Stage 4
    ###############
    stage4_args = {}

    output, output_scale = tensor_quant_scale(output, bits=8)
    stage3_args['M_stage3'] = (1/output_scale)
    output = output.type(torch.int8)
    dense_weight, dense_bias, dense_acc_scale = quantize_linear_params(layer.output.dense, output_scale)
    stage4_args['dense_weight_t'] = dense_weight.T
    stage4_args['dense_bias'] = dense_bias
    stage4_args['dense_acc_scale'] = dense_acc_scale
    stage4_args['skip_conn'] = attention_out
    stage4_args['layernorm'] = layer.output.LayerNorm
    output = linear_kernel(output, dense_weight.T, dense_bias)

    # DEQUANT ZONE BEGIN ⬇️`
    output = output.float() * dense_acc_scale

    output = tensor_quant_layernorm(layer.output.LayerNorm, output + attention_out)

    _, output_scale = tensor_quant_scale(output, bits=8)
    stage4_args['M_stage4'] = (1/output_scale)



    ### BY THE PIPELINE
    stage1_out = stage1(**stage1_args)
    stage2_out = stage2(*stage1_out, **stage2_args)
    stage3_out = stage3(stage2_out, **stage3_args)
    stage4_out = stage4(stage3_out, **stage4_args)

    output = stage4_out.float() * output_scale
    return output


def attention(layer, hidden_states, attention_mask=None):
    '''
    Pass in a encoder layer (which holds pretrained weights) and hidden_states input,
    and this function performs the same operations as the layer but in a readable fashion.
    
    hidden_states: <bs, seqlen, dmodel>
    '''
    bs, seqlen, dmodel = hidden_states.size()
    num_heads = layer.attention.self.num_attention_heads
    dhead = layer.attention.self.attention_head_size
    
    # Linear transform to get multiple heads. This is a major MAC slurper.
    # Each of these is calling an nn.Linear layer on hidden_states.
#     query_layer = layer.attention.self.query(hidden_states) # 
    query_layer = tensor_quant_linear(layer.attention.self.query, hidden_states) # <bs, seqlen, dmodel>
    key_layer = tensor_quant_linear(layer.attention.self.key, hidden_states)     # <bs, seqlen, dmodel>
    value_layer = tensor_quant_linear(layer.attention.self.value, hidden_states) # <bs, seqlen, dmodel>

    # Reshape and transpose for multi-head
    new_shape = (bs, seqlen, num_heads, dhead)
    
    query_layer = query_layer.view(new_shape)
    value_layer = value_layer.view(new_shape)
    key_layer = key_layer.view(new_shape)
    
    query_layer = query_layer.permute(0,2,1,3) # <bs, num_head, seqlen, dhead>
    value_layer = value_layer.permute(0,2,1,3) # <bs, num_head, seqlen, dhead>
    # Key is transposed to match dimensions of Query for matmul
    key_layer = key_layer.permute(0,2,3,1)     # <bs, num_head, dhead, seqlen>
    
    # The attention main course
    attention_scores = tensor_quant_matmul(query_layer, key_layer)
    attention_scores /= math.sqrt(dhead)
    
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
        attention_scores = attention_scores + attention_mask
    
    attention_probs = tensor_quant_softmax(attention_scores)
    # attention_probs = layer.attention.self.dropout(attention_probs)
    
    # Weighted sum of Values from softmax attention
    attention_out = tensor_quant_matmul(attention_probs, value_layer)
    
    attention_out = attention_out.permute(0,2,1,3).contiguous()
    attention_out = attention_out.view(bs, seqlen, dmodel)
    
    # It's time for one more linear transform and layer norm
    dense_out = tensor_quant_linear(layer.attention.output.dense, attention_out)
    # dense_out = layer.attention.output.dropout(dense_out)
    
    # LayerNorm also mplements the residual connection
    layer_out = tensor_quant_layernorm(layer.attention.output.LayerNorm, dense_out + hidden_states)
    
    return layer_out


def ffn(layer, attention_out):
    '''
    Pass in the feedforward layer and attention output. Returns the same result of a FF forward pass.
    '''
    # Layer 1
    output = tensor_quant_linear(layer.intermediate.dense, attention_out)
    output = tensor_quant_gelu(output)
    
    # Layer 2
    output = tensor_quant_linear(layer.output.dense, output)
    # output = layer.output.dropout(output)
    output = tensor_quant_layernorm(layer.output.LayerNorm, output + attention_out)
    
    return output


def encoder(model, hidden_states, attention_mask):
    '''
    Pass input through encoder stack
    '''
    for layer_module in model.encoder.layer:
        # MHA + LayerNorm
        # attention_out = attention(layer_module, hidden_states, attention_mask)
        # hidden_states = ffn(layer_module, attention_out)
        hidden_states = layer_kernel(layer_module, hidden_states, attention_mask)

    return hidden_states