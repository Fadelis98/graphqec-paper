# GraphQEC

Python package for neural-network decoding of stabilizer-based quantum error correction codes, as presented in [Efficient and Universal Neural-Network Decoder for Stabilizer-Based Quantum Error Correction](https://arxiv.org/abs/2502.19971).

This github repo focus on providing the nessary data and code to reproduce the results in the paper. If you are interested in developing your project or training your own model, keep track of [Graphqec-lib](https://github.com/Fadelis98/GraphQEC-lib). We are working on refactoring the codebase to provide a more user-friendly interface for training and benchmarking.

## Features

- **Supported Codes**:
  - [Sycamore Surface Codes](https://doi.org/10.5281/zenodo.6804040)
  - [Color Codes](https://github.com/seokhyung-lee/color-code-stim)
  - [BB Codes](https://github.com/gongaa/SlidingWindowDecoder)
- **Integrated Decoders**:
  - [BPOSD](https://github.com/quantumgizmos/ldpc)
  - [PyMatching](https://github.com/oscarhiggott/PyMatching)
  - [Concatenated Matching](https://github.com/seokhyung-lee/color-code-stim)
- **Neural network decoders**
  - [GraphRNNDecoderV5A]: A pure RNN version, excellent for small-scale codes.
  - [GraphLinearAttnDecoderV2A]: Linear attention version used in the paper, easier to train for large codes.

## install

### from pip
```bash
# Base installation
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e .

# For CUDA 11.8:
# pip install --extra-index-url https://download.pytorch.org/whl/cu118 -e .
```

Note: Choose the PyTorch version matching your CUDA version. See PyTorch installation guide for more options.

## Pretrained Models
Download trained weights from the [Releases page](https://github.com/Fadelis98/graphqec-paper/release)

## Usage

The `graphqec.benchmark.evaluate` submodule contains several functions that can benchmark the results on a slurm cluster automatically.

```python
import datetime
from graphqec.benchmark import evaluate as evl

slurm_path = f"./tmp/slurm/{datetime.datetime.now().strftime('%Y-%m-%d')}"
test_configs = json.load(open("configs/benchmark/graphqec_acc/BB72.json",'r'))
results = submit_benchmark(slurm_path,test_configs,debug=True)
```

The `test_decoder.ipynb` in the root directory also contains a minimal example of how to decode a code using the neural network decoder.

## kown problems

### causal-conv1d package
`causal-conv1d` may not compatible with some device. It will cause an error when running the linear attention version of neural network decoder:
```
RuntimeError: Please either install causal-conv1d>=1.4.0 to enable fast causal short convolution CUDA kernel or set use_fast_conv1d to False.
```
Turnning of `use_fast_conv1d` will not fix this problem. Please view [their repo](https://github.com/Dao-AILab/causal-conv1d/tree/main) to check the compatibility.

### `torch.compile` support

Some versions of `flash-linear-attention` missed `@torch.compiler.disable` decorator before their trition kernels. If `torch.compile` failed, check whether the decorator is applied to the `fused_recurrent_gated_delta_rule` kernel in `fla/ops/delta_rule/fused_recurrent`.


## Citation

```bibtex
@article{hu2025efficient,
  title={Efficient and Universal Neural-Network Decoder for Stabilizer-Based Quantum Error Correction},
  author={Hu, Gengyuan and Ouyang, Wanli and Lu, Chao-Yang and Lin, Chen and Zhong, Han-Sen},
  journal={arXiv preprint arXiv:2502.19971},
  year={2025}
}
```