# trans-fat
An FPGA Accelerator for Transformer Inference

[huggingface Bert implementation](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py) is our reference implementation. The source code is easy to read and follow along with.

## Python Implementation
Quantized operations are implemented in `bert_sw/src/quant_ops.py`. These use dynamic quantization, which can be mapped to static quantization if you program in the input range.

Quantized BERT layer is implemented in `bert_sw/src/quant_layer.py`. This implements attention and ffn using `quant_ops` in the simplest possible way.

Quantized BERT (RoBERTa) implementation is in `bert_sw/src/quant_roberta.py`. This makes it easy to test end-task accuracy with a quantized encoder.

Quantized RoBERTa accuracy test is in `bert_sw/test_quant_roberta.ipynb`. Run this to ensure accuracy is high after modifying quantized operations (91% is unquantized validation accuracy).

## Instructions
TODO

## Optimization Versions

### v0
- None

### v1
- Linear layer tiling
- Buffering of input and output data
- Unrolling of multiplication inner loops

### v2
- Transpose A matmul input
- Cache line of A.T
- Increase tile size in j dimension 
- Unrolling of computation in attention heads

### v3
- Stream only in linear layers
- stage1 read A outside linear_fused. Write to A streams. Have each dense matmul read from an A stream and compute its product in parallel

## Results

<table align="center">
<thead>
  <tr>
    <th rowspan="2">Version</th>
    <th colspan="3">Latency (ms)</th>
  </tr>
  <tr>
    <th>fpga1</th>
    <th>fpga2</th>
    <th>all</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">v0</td>
    <td>4723.71</td>
    <td>10950.90</td>
    <td>15676.30</td>
  </tr>
  <tr>
    <td align="center">v1</td>
    <td>274.98</td>
    <td>120.91</td>
    <td>397.45</td>
  </tr>
  <tr>
    <td align="center">v2</td>
    <td>48.36</td>
    <td>95.60</td>
    <td>145.27</td>
  </tr>
  <tr>
    <td align="center">v3</td>
    <td>35.03</td>
    <td>71.76</td>
    <td>110.99</td>
  </tr>
</tbody>
</table>


