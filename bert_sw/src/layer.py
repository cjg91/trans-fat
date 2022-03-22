import torch
import torch.nn as nn
import math

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
#     query_layer = layer.attention.self.query(hidden_states) # <bs, seqlen, dmodel>
    query_layer = layer.attention.self.query(hidden_states)
    key_layer = layer.attention.self.key(hidden_states)     # <bs, seqlen, dmodel>
    value_layer = layer.attention.self.value(hidden_states) # <bs, seqlen, dmodel>
    
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
    attention_scores = torch.matmul(query_layer, key_layer)
    attention_scores /= math.sqrt(dhead)
    
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
        attention_scores = attention_scores + attention_mask
    
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    # attention_probs = layer.attention.self.dropout(attention_probs)
    
    # Weighted sum of Values from softmax attention
    attention_out = torch.matmul(attention_probs, value_layer)
    
    attention_out = attention_out.permute(0,2,1,3).contiguous()
    attention_out = attention_out.view(bs, seqlen, dmodel)
    
    # It's time for one more linear transform and layer norm
    dense_out = layer.attention.output.dense(attention_out)
    # dense_out = layer.attention.output.dropout(dense_out)
    
    # LayerNorm also mplements the residual connection
    layer_out = layer.attention.output.LayerNorm(dense_out + hidden_states)
    
    return layer_out


def ffn(layer, attention_out):
    '''
    Pass in the feedforward layer and attention output. Returns the same result of a FF forward pass.
    '''
    # Layer 1
    output = layer.intermediate.dense(attention_out)
    output = nn.functional.gelu(output)
    
    # Layer 2
    output = layer.output.dense(output)
    # output = layer.output.dropout(output)
    output = layer.output.LayerNorm(output + attention_out)
    
    return output


def encoder(model, hidden_states, attention_mask):
    '''
    Pass input through encoder stack
    '''
    for layer_module in model.encoder.layer:
        # MHA + LayerNorm
        attention_out = attention(layer_module, hidden_states, attention_mask)
        ff_out = ffn(layer_module, attention_out)
        hidden_states = ff_out

    return hidden_states