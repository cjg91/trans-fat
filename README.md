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
- buffering of input and output data
- unrolling of multiplication inner loops

### v2
- Transpose A matmul input
- Cache line of A.T
- Increase tile size in j dimension 
- Buffer data between stages on-chip

### v3
- Stream
- TODO: Remove unnecesasry rowbuf from stage4 layernorm

