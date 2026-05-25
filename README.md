# SIKA-GP

[ICML 2026] **SIKA-GP: Accelerating Gaussian Process Inference with Sparse Inducing Kernel Approximations for Bayesian Deep Learning**

SIKA-GP is a PyTorch/GPyTorch codebase for accelerating Gaussian process (GP) inference using Sparse Inducing Kernel Approximations. SIKA-GP builds sparsely activated Laplace-kernel basis functions on a dyadic ordered inducing grid. Each input activates only about `O(log M)` basis functions, replacing dense inducing-kernel computation in GP, Deep Kernel Learning, and Deep Gaussian Process models with tensorized sparse indexing.


## Environment Setup

From the repository root, run:

```bash
conda env create -f environment.yml
conda activate sika-gp
```

## Quick Start

### 1. Use a SIKA-GP Layer

```python
import torch
from csgp.layers.gps import CSGP

x = torch.randn(32, 16)
layer = CSGP(in_features=16, out_features=10, dyadic_level=7)
logits, kl = layer(x, return_kl=True, sparse=True)
print(logits.shape, kl)
```

`dyadic_level=L` corresponds to about `2^L - 1` internal inducing basis functions. In sparse mode, each feature dimension uses only a small active subset, which is useful for high-dimensional deep-feature settings.

### 2. Time-Complexity Experiment

```bash
python examples/time_analysis.py \
  --log-dir ./logs/time_analysis \
  --batch-size 128 \
  --in-features 128 \
  --out-features 128 \
  --samples 10
```

### 3. MNIST Image Classification

Run a small smoke test:

```bash
python examples/mnist/run_mnist.py \
  --mode train \
  --model dak \
  --epochs 20 \
  --subset-size 2000 \
  --batch-size 128 \
  --num_mc_train 2 \
  --num_mc_test 5
```

Test a saved checkpoint:

```bash
python examples/mnist/run_mnist.py \
  --mode test \
  --model dak \
  --batch-size 128 \
  --num_mc_test 20
```

Available model choices include `nn`, `svdkl`, and `dak`. The `dak` option uses the SIKA/DAK-style Bayesian GP head implemented in this repository.

### 4. CIFAR-10/100 Image Classification

```bash
python examples/cifar/run_cifar.py \
  --mode train \
  --model dak \
  --num_classes 10 \
  --arch resnet18 \
  --epochs 1 \
  --batch-size 128 \
  --num_mc_train 2
```

Test:

```bash
python examples/cifar/run_cifar.py \
  --mode test \
  --model dak \
  --num_classes 10 \
  --arch resnet18 \
  --batch-size 128
```

Use `--num_classes 10` for CIFAR-10 and `--num_classes 100` for CIFAR-100. The script downloads torchvision datasets automatically and writes logs and checkpoints to `logs/` and `checkpoint/` under the current working directory.

### 5. UCI Regression

UCI datasets are not downloaded automatically. Place data files under:

```text
examples/uci/datasets/<dataset_name>/data.csv.gz
```

The expected format follows `treforevans/uci_datasets`: each row is comma-separated numeric data, and the last column is the regression target.

Run a quick experiment:

```bash
python examples/uci/run_uci.py \
  --running-datasets parkinsons \
  --model-names dgp-sparse dgp-dense \
  --epochs 1 \
  --kfolds 2 \
  --batch-sizes 128 \
  --nn-out-features 16
```

### 6. CLINC150 OOD and Transformer Experiments

This experiment downloads the `clinc_oos` dataset and `distilbert-base-uncased` weights from Hugging Face, so network access is required.

```bash
cd examples/llm
python clinic150_ood.py \
  --bayes csgp \
  --epochs 1 \
  --batch_size 16 \
  --T 5 \
  --device cuda \
  --log_csv clinic150_ood_csgp_runs.csv
```

Run multiple seeds:

```bash
cd examples/llm
python run_clinic150_ood_seeds.py \
  --bayes csgp \
  --seeds 0 1 2 \
  --epochs 3 \
  --batch_size 16 \
  --T 10
```

The `--bayes` option can be `csgp`, `bayes_linear`, or `svdkl`.


## Notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

Main notebooks:

- `examples/1D_sika_gp.ipynb`: SIKA-GP/DGP visualization on 1D regression.
- `examples/1D_induced_feature.ipynb`: inducing features and GP-basis examples.
- `examples/compact_support.ipynb`: compact support and dyadic-basis structure.
- `examples/lightweight_linear.ipynb`: lightweight sparse-linear indexing experiments.


## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{zhao2026sika,
  title     = {SIKA-GP: Accelerating Gaussian Process Inference with Sparse Inducing Kernel Approximations for Bayesian Deep Learning},
  author    = {Zhao, Wenyuan and Tuo, Rui and Tian, Chao},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```

## License

This project is released under the MIT License.
