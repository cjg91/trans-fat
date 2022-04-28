# trans-fat
An FPGA Accelerator for Transformer Inference

We accelerated a BERT layer across two FPGAs, partitioned into four pipeline stages. We conduct three levels of optimization using Vitis HLS and report runtimes. The accelerator implements a transformer layer of standard BERT size, with a sequence length of 128 (which can be modified).

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
- Stream DDR inputs/outputs in linear layers

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


