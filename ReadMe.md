# DualPriorGT: Dual-Prior Graph Transformer for Few-Shot HSI Classification

Official implementation of **SAM-DPGT: SAM-Guided Dual-Prior Graph Transformer for Few-Shot Hyperspectral Image Classification**.

## Architecture

```
Raw HSI [H, W, C]
    |
    v
CNN Head (1x1 Conv + BN + LeakyReLU) --> H_pixel [H*W, 128]
    |
    v  (SAM superpixel pooling via S_hat^T)
H_node [N_sp, 128]
    |
    v  (Linear projection + LeakyReLU)
H_node [N_sp, 128]
    |
    v  (Dual-Prior Attention x L blocks)
H_node [N_sp, 128]
    |
    v  (Superpixel-to-pixel upsampling via S)
H_up [H*W, 128]
    |
    v  (Skip concat: [H_up, H_pixel])
H_combined [H*W, 256]
    |
    v  (CNN Tail: BN + 1x1 Conv + 5x5 DWConv)
H_out [H*W, 128]
    |
    v  (Classifier: Linear + Softmax)
Predictions [H*W, n_classes]
```

### Dual-Prior Attention (DPAttn)

Each DPAttn block computes:

1. **Content similarity**: `E = Q K^T / sqrt(d)` where `Q = K = V = W_shared(H)`
2. **k-hop mask**: `M = ((A + I)^k > 0)` — binary connectivity mask
3. **RBF spatial bias**: `R_jk = sum_c w_c * exp(-(D_jk - mu_c)^2 / (2 sigma^2))`
4. **Fusion**: `E_hat = E + alpha * I + beta * R` (learnable self-loop alpha, learnable per-head spatial weight beta)
5. **Masked softmax**: `A = softmax(E_tilde / temperature)` where `E_tilde = E_hat * M` (masked)
6. **Output**: `O = W_out(Concat(heads))` followed by LeakyReLU

## Requirements

```
torch >= 1.12
numpy
scipy
scikit-learn
scikit-image
segment_anything
h5py
matplotlib
spectral
```

Install SAM:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download SAM checkpoint (ViT-B):
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Datasets

Place datasets in the `data/` directory:

```
data/
├── indian/
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── paviaU/
│   ├── PaviaU.mat
│   └── Pavia_University_gt.mat
├── Salinas/
│   ├── Salinas_corrected.mat
│   └── Salinas_gt.mat
└── TeaFarm/
    ├── Teafarm.mat
    └── Teafarm_gt.mat
```

## Usage

```bash
python run_final.py
```

### Configuration

Edit the parameters at the top of `run_final.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_HOP` | 1 | k-hop connectivity for mask M |
| `N_TRAIN` | 5 | Training samples per class |
| `N_VAL` | 5 | Validation samples per class |
| `N_SEEDS` | 10 | Number of random seeds |
| `NUM_HEADS` | 4 | Attention heads |
| `BASE_CHANNELS` | 128 | Feature dimension |
| `LR` | 5e-4 | Learning rate (AdamW) |
| `MAX_EPOCHS` | 301 | Maximum training epochs |

## File Structure

```
├── DualPrior_GT_Github.py   # Model definition
├── utils.py                  # Utility functions
├── run_final.py              # Main training script
├── requirements.txt          # Dependencies
├── results/                  # Output results (auto-created)
└── data/                     # Dataset directory (set DATA_ROOT)
```

## Results

Precomputed per-seed results are provided in `results/` (numbers only, no figures):

- `results/results.json` — per-seed OA/AA/Kappa (and per-class accuracies where available) of the 10-seed runs reported in the paper (5 training + 5 validation labels per class). The mean ± std values correspond to the paper's main tables.
- `results/results_rerun_validation.json` — an independent rerun of the full model using this released code (seeds 0–9, same protocol), verifying that the released pipeline reproduces the paper results within run-to-run variation.

## Citation

```bibtex
@article{liu2024samdpgt,
  title={SAM-DPGT: SAM-Guided Dual-Prior Graph Transformer for Few-Shot Hyperspectral Image Classification},
  author={Liu, Qichao and Xiao, Liang and Huang, Nan and Tang, Jinhui},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
```

## License

This project is released under the MIT License.
