# trans-fat
An FPGA Accelerator for Transformer Inference

## Bert Layer
`./single_layer.ipynb` notebook demonstrates the operations within one single Bert layer.

`./int8_bert.ipynb` generates ground truth results of a quantized encoder.

[huggingface Bert implementation](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py) is our reference implementation. The source code is easy to read and follow along with.

