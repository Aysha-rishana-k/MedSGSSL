<![CDATA[<div align="center">

# 🫁 MedSGSSL

**Medical Segmentation-Guided Self-Supervised Learning for Chest X-Ray Classification**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Leveraging rule-based lung segmentation priors to improve self-supervised representation learning on chest X-rays, without requiring any segmentation labels.*

</div>

---

## 📖 Overview

MedSGSSL explores **7 distinct approaches** to integrating lung segmentation information into self-supervised learning (SSL) pipelines for multi-label chest X-ray disease classification. Instead of relying on expensive annotated segmentation masks, we use **rule-based lung segmentation** (adaptive thresholding + morphological operations) to generate anatomical priors that guide the SSL process.

Each option is evaluated against a **direct classification baseline** (no SSL pretraining) to measure the benefit of self-supervised pretraining with segmentation guidance.

### Key Features

- 🫁 **No segmentation labels needed** — uses rule-based lung masks (adaptive thresholding, morphology)
- 🔄 **7 SSL strategies** — from contrastive learning to masked attention modeling
- 📊 **14 disease classification** — multi-label classification on NIH ChestX-ray14
- ⚡ **Optimized pipeline** — cv2 image loading, precomputed mask caching, fast augmentations
- 🏥 **Patient-level splits** — no data leakage between train/val/test

---

## 🏗️ Architecture Options

### SSL Approaches

| Option | Strategy | Encoder | Key Idea |
|--------|----------|---------|----------|
| **1** | Masked Contrastive | MobileNetV2 | Lung-aware masking + NT-Xent contrastive loss |
| **2** | Multi-Region | MobileNetV2 | Separate lung/non-lung region encodings + region-contrastive loss |
| **3** | Adaptive Pathology | MobileNetV2 | Pathology-weighted contrastive loss based on segmentation regions |
| **4** | Segmentation-Guided Crop | MobileNetV2 | Context-aware lung cropping + contrastive SSL |
| **5** | Attention Segmentation | MobileNetV2 | Attention-guided feature extraction with segmentation priors |
| **6** | Segmentation Channel | MobileNetV2 | 2-channel input (image + segmentation mask) |
| **7** | Anatomy-Masked Attention | ViT-Small | Pretrained ViT + masked reconstruction + attention alignment + contrastive loss |

### Pipeline

```
                                    ┌─────────────────────┐
                                    │   NIH ChestX-ray14  │
                                    │   112,120 images    │
                                    └────────┬────────────┘
                                             │
                                    ┌────────▼────────────┐
                                    │  Precompute Lung    │
                                    │  Masks (Rule-Based) │
                                    └────────┬────────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          │                  │                  │
                   ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
                   │  SSL Pretrain│   │  SSL Pretrain│   │     ...     │
                   │  (Option 1) │   │  (Option 7) │   │ (Options 2-6│
                   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                          │                  │                  │
                   ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
                   │  Fine-tune  │   │  Fine-tune  │   │  Fine-tune  │
                   │  Classifier │   │  Classifier │   │  Classifier │
                   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                          │                  │                  │
                          └──────────────────┼──────────────────┘
                                             │
                                    ┌────────▼────────────┐
                                    │  14-Disease AUC     │
                                    │  Evaluation         │
                                    └─────────────────────┘
```

---

## 📁 Project Structure

```
MedSGSSL/
├── precompute_lung_masks.ipynb                        # Step 0: Generate lung masks
│
├── option1_masked_contrastive_ssl.ipynb                # Option 1: SSL
├── option1_masked_contrastive_direct_classification.ipynb  # Option 1: Baseline
│
├── option2_multi_region_ssl.ipynb                      # Option 2: SSL
├── option2_multi_region_direct_classification.ipynb     # Option 2: Baseline
│
├── option3_adaptive_pathology_ssl.ipynb                # Option 3: SSL
├── option3_adaptive_pathology_direct_classification.ipynb  # Option 3: Baseline
│
├── option4_segmentation_guided_crop_ssl.ipynb          # Option 4: SSL
├── option4_segmentation_guided_crop_direct_classification.ipynb  # Option 4: Baseline
│
├── option5_attention_segmentation_ssl.ipynb             # Option 5: SSL
├── option5_attention_segmentation_direct_classification.ipynb  # Option 5: Baseline
│
├── option6_ssl_segmentation_channel.ipynb              # Option 6: SSL
├── option6_segmentation_channel_direct_classification.ipynb  # Option 6: Baseline
│
├── option7_anatomy_masked_attention_ssl.ipynb           # Option 7: SSL (ViT-Small)
│
├── baseline_ssl_chest_xray.ipynb                       # Vanilla SSL baseline
├── COMPLETE_IMPLEMENTATION_GUIDE.md                    # Detailed technical guide
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision timm opencv-python pandas scikit-learn tqdm matplotlib seaborn
```

### Dataset

Download the [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) dataset:

```
data/
├── images/                    # All 112,120 chest X-ray images
├── Data_Entry_2017.csv        # Image labels and metadata
└── precomputed_masks/         # Generated by Step 0 (below)
```

### Step 0: Precompute Lung Masks

Run `precompute_lung_masks.ipynb` to generate rule-based lung segmentation masks for all images. These masks are saved to disk and used by all subsequent notebooks.

### Step 1: Choose an Option and Run

Each option has two notebooks:
- **`*_ssl.ipynb`** — SSL pretraining → fine-tuning → evaluation
- **`*_direct_classification.ipynb`** — Direct classification baseline (no SSL)

Run any option's SSL notebook end-to-end. The notebook handles:
1. Data loading with precomputed masks
2. Patient-level train/val/test split
3. SSL pretraining
4. Supervised fine-tuning
5. Per-disease AUC evaluation

---

## 🔬 Option Details

### Option 1: Masked Contrastive Learning
Masks lung regions in chest X-rays and uses NT-Xent contrastive loss to learn representations that are invariant to which lung patches are visible. Forces the model to understand the full lung anatomy from partial views.

### Option 2: Multi-Region Contrastive
Encodes lung and non-lung regions separately, then applies region-contrastive loss to learn discriminative features for anatomical regions.

### Option 3: Adaptive Pathology-Weighted
Applies pathology-aware weighting to the contrastive loss based on segmentation regions, so the model focuses more on disease-relevant areas.

### Option 4: Segmentation-Guided Cropping
Uses lung bounding boxes for context-aware cropping, then applies contrastive SSL on the cropped views. Ensures the model always sees relevant anatomy regardless of image framing.

### Option 5: Attention-Guided Segmentation
Combines attention mechanisms with segmentation priors to guide feature extraction, learning to focus on anatomically relevant regions.

### Option 6: Segmentation as Input Channel
Concatenates the lung segmentation mask as a second input channel (2-channel: image + mask), directly providing anatomical information to the network.

### Option 7: Anatomy-Constrained Masked Attention (ViT)
The most advanced approach — uses a **pretrained ViT-Small** encoder with:
- **Masked reconstruction** with a momentum teacher (DINO/MAE-inspired)
- **Attention alignment** — CLS attention guided towards lung regions
- **Contrastive loss** — NT-Xent between student and teacher representations
- **Encoder freezing** — pretrained weights frozen for initial epochs, then unfrozen

---

## ⚙️ Configuration

Each notebook contains a `Config` class with tunable hyperparameters:

```python
class Config:
    img_size = 224
    batch_size = 64
    pretrain_epochs = 50
    finetune_epochs = 50
    lr_pretrain = 1e-4
    lr_finetune = 1e-4
    num_workers = 4              # Auto-adjusted for available CPUs
    cache_masks = True           # Pre-load masks into RAM
    
    # Option 7 specific
    use_pretrained = True        # ImageNet pretrained ViT-Small
    freeze_epochs = 5            # Freeze encoder for N initial SSL epochs
    lambda_attn = 0.05           # Attention alignment loss weight
    lambda_contrastive = 0.1     # Contrastive loss weight
```

---

## 📊 Evaluation

All options are evaluated on **14 thoracic diseases** from the NIH ChestX-ray14 dataset:

> Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia

**Metrics:**
- Per-disease AUC-ROC
- Mean AUC across all diseases
- Per-disease F1 score with optimal thresholds
- Precision and Recall

---

## 🛠️ Performance Optimizations

All notebooks include the following optimizations for fast training:

| Optimization | Description |
|---|---|
| **cv2 image loading** | `cv2.imread` instead of `PIL.Image.open` (~2x faster) |
| **Precomputed masks** | Rule-based masks computed once, loaded from disk |
| **Mask RAM caching** | Optional `cache_masks=True` to eliminate per-sample disk I/O |
| **Fast augmentations** | `cv2.warpAffine` rotations (no scipy dependency) |
| **Multi-worker loading** | `num_workers` auto-detected from CPU count |
| **Pinned memory** | `pin_memory=True` for faster CPU→GPU transfer |
| **Checkpoint resume** | Training resumes from latest checkpoint on restart |

---

## 🐳 Docker Support

When running in Docker, ensure sufficient shared memory for PyTorch DataLoader workers:

```bash
docker run --shm-size=8g --gpus all -v /path/to/data:/data your-image
```

The notebooks auto-detect insufficient `/dev/shm` and warn accordingly.

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{medsgssl2025,
  title={MedSGSSL: Segmentation-Guided Self-Supervised Learning for Chest X-Ray Classification},
  author={Aysha Rishana K},
  year={2025},
  url={https://github.com/Aysha-rishana-k/MedSGSSL}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<i>Built with ❤️ for medical AI research</i>
</div>
]]>
