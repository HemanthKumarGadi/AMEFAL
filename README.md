# AMEFAL — Attention-Guided Multi Embedding Fusion based Active Learning Framework for Satellite Image Segmentation

> **UNet + DINOv2-LoRA + TinyWeightMLP** — an active learning pipeline that combines entropy uncertainty, UNet embedding diversity, and DINOv2 embedding diversity, fused by a learned meta-weight MLP, to iteratively label the most informative satellite images.

---


**Key components:**
- **UNet** with a bottleneck embedding head (embedding_dim = 256)
- **DINOv2** (`facebook/dinov2-base`) with **LoRA** fine-tuning (rank=4, alpha=4, injected into attention layers 10–11)
- **TinyWeightMLP** — 3-input MLP (hidden=32) that predicts per-sample fusion weights, trained online using validation mIoU improvement as a reward
- **DINOv2 is re-fine-tuned every 4 iterations** as the labelled set grows

**9 segmentation classes:** Background, Bareland, Rangeland, Developed Space, Road, Tree, Water, Agriculture Land, Building

---

## Prerequisites

| Requirement | Minimum |
|---|---|
| Python | 3.9+ |
| CUDA | 11.8 or 12.1 (GPU required — CPU not supported) |
| GPU VRAM | 8 GB+ (single GPU) / 2 × 8 GB (multi-GPU) |
| RAM | 16 GB+ |
| Disk space | ~15 GB (DINOv2 cache + checkpoints + outputs) |

> **Note:** The script will raise a `RuntimeError` at startup if no CUDA GPU is detected.

```


```
<BASE_DIR>/
│
├── train_data/           ← Initial labelled training images
├── train_labels/         ← Corresponding segmentation masks (same filenames)
│
├── Unlabeled_data/       ← Unlabelled pool — images selected from here each iteration
├── Validation_labels/    ← Labels for pool images (revealed after selection)
│
├── val_img/              ← Validation images (fixed throughout all iterations)
├── val_lab/              ← Validation masks
│
├── test_img/             ← Test images (evaluated every iteration)
└── test_lab/             ← Test masks
```

### Mask format

| Rule | Detail |
|---|---|
| Format | Grayscale PNG, JPG, or TIF |
| Pixel values | Integer class indices **0–8** only |
| Filename pairing | `image_001.png` must have `image_001.png` in the labels folder |
| Resolution | Any — all images are resized to **512 × 512** internally |

### Class index reference

| Index | Class |
|---|---|
| 0 | Background |
| 1 | Bareland |
| 2 | Rangeland |
| 3 | Developed Space |
| 4 | Road |
| 5 | Tree |
| 6 | Water |
| 7 | Agriculture Land |
| 8 | Building |

---

## Configuration

Open `AAMEAL_Final_Code_with_Biplot.ipynb` (or the exported `.py`) and locate the `main()` function. Edit these values before running:

```python
# ── Base path ─────────────────────────────────────────────────────────────
BASE_DIR    = r"E:\hemanth\data\data"     # ← change to your data root
RESULTS_DIR = os.path.join(BASE_DIR, "updated_plots__01-04")  # ← output folder

# ── Active learning ────────────────────────────────────────────────────────
MAX_ITERATIONS          = 20    # total AL rounds
SAMPLES_PER_ITERATION   = 50    # pool images moved to training each round
DINO_FINETUNE_INTERVAL  = 4     # re-finetune DINOv2 every N iterations

# ── Training schedule ──────────────────────────────────────────────────────
EPOCHS_FIRST_ITERATION  = 100   # iteration 0: full run, no early stopping
EPOCHS_SUBSEQUENT_MAX   = 50    # iterations 1+: max epochs, early-stop enabled
EARLY_STOPPING_PATIENCE = 10    # patience for early stopping

# ── Model & optimiser ─────────────────────────────────────────────────────
BATCH_SIZE     = 4              # reduce to 2 if VRAM < 8 GB
LEARNING_RATE  = 0.0001
EMBEDDING_DIM  = 256            # UNet latent space dimension

# ── DINOv2 ────────────────────────────────────────────────────────────────
DINOV2_MODEL   = 'facebook/dinov2-base'   # or dinov2-small / dinov2-large
LORA_RANK      = 4              # fixed — do not change between iterations
LORA_ALPHA     = 4.0            # LoRA scale = alpha / rank = 1.0

# ── Diversity ─────────────────────────────────────────────────────────────
DIVERSITY_METRIC = 'cosine'     # 'cosine' or 'euclidean'
```

---

## Running the Pipeline

### Option A — Run the Jupyter notebook

```bash
jupyter notebook AAMEAL_Final_Code_with_Biplot.ipynb
```

Run all cells from top to bottom. The final cell calls `main()`.

### Option B — Export and run as a Python script

```bash
# Export notebook to script
jupyter nbconvert --to script AAMEAL_Final_Code_with_Biplot.ipynb

# Run
python AAMEAL_Final_Code_with_Biplot.py
```

---

## Interactive Prompts — What to Enter

The pipeline asks four questions at startup. Here is exactly what each prompt expects:

---

### Prompt 1 — GPU Selection

```
Available GPUs: 1
  GPU 0: NVIDIA GeForce RTX 4070 Ti

Select GPU mode  (1 = Single GPU,  2 = Multi-GPU):
```

| Input | Effect |
|---|---|
| `1` | Use GPU 0 only ← recommended for single-GPU machines |
| `2` | DataParallel across all GPUs (requires ≥ 2 GPUs) |

---

### Prompt 2 — Sampling Methods

```
ACTIVE LEARNING SAMPLING METHOD SELECTION
  1 → Entropy only
  2 → UNet Diversity only
  3 → DINOv2 Diversity only
  1,2 → Entropy + UNet Diversity
  1,3 → Entropy + DINOv2 Diversity
  2,3 → UNet + DINOv2 Diversity
  1,2,3 → All three (Entropy + UNet Div + DINOv2 Div)  ← RECOMMENDED
```


> **Tip:** Use `7` for the full AAMEAL framework. Use `1` for a quick baseline.








---

### Prompt 3 — DINOv2 Fine-tuning Method

```
DINOV2 FINE-TUNING METHOD SELECTION
  Training type:  1 = Supervised   2 = Self-Supervised
  Fine-tune method: 1 = Multi-Label   2 = LoRA
```

**Recommended combination:**

| Step | Input | Reason |
|---|---|---|
| Training type | `1` (Supervised) | Uses your labelled masks directly |
| Fine-tune method | `2` (LoRA) | Only 2 × 4 projection layers trained — stable warm-start across iterations |

> Self-supervised (`2`) trains DINOv2 on masked patch reconstruction — useful when labelled data is very scarce (< 50 samples).

---

### Prompt 4 — Visualizations

```
VISUALIZATION SELECTION
  1 → Uncertainty Map  (Decision Margin: Original | GT | Pred | Margin)
  2 → Saliency Map     (Confidence:      Original | GT | Pred | MaxSoftmax)
  3 → Embedding Plots  (PCA + t-SNE per split: Train / Pool / Test)
  4 → PAE Map          (ProxGradCAM:     Original | Depth | Depth-inf | CAM)

  Enter numbers comma-separated, e.g. '1,2,3,4'
```

---

## Outputs

All outputs are written to `RESULTS_DIR` (default: `<BASE_DIR>/updated_plots__01-04/`).

```
RESULTS_DIR/
│
├── training_summary.csv          ← One row per iteration: all metrics
├── mlp_weight_history.csv        ← MLP-predicted method weights per iteration
├── checkpoint.pth                ← Full checkpoint for resuming
├── finetuned_dinov2.pth          ← Saved DINOv2 weights (updated every 4 iters)
├── dinov2_cache/                 ← Hugging Face model cache (downloaded once)
│
└── iteration_N/
    ├── model_iteration_N.pth                    ← UNet weights
    ├── detailed_metrics_iteration_N.csv         ← Per-class IoU + all metrics
    │
    ├── test_predictions/                        ← Prediction panels (all test images)
    │   └── sample_K_iter_N.png                 ← Image | GT | Pred
    │
    ├── uncertainty_maps/                        ← [if option 1 selected]
    │   └── uncertainty_sample_K_iter_N.png     ← Image | GT | Pred | Decision Margin
    │
    ├── saliency_maps/                           ← [if option 2 selected]
    │   └── saliency_sample_K_iter_N.png        ← Image | GT | Pred | Grad-CAM
    │
    ├── embedding_plots/                         ← [if option 3 selected]
    │   ├── pca_embeddings_iter_N.png            ← PCA scatter (Train/Pool/Test)
    │   ├── tsne_embeddings_iter_N.png           ← t-SNE scatter
    │   └── pca_biplot_iter_N.png                ← ggplot2-style PCA biplot
    │
    ├── pae_maps/                                ← [if option 4 selected]
    │   └── pae_sample_K_iter_N.png             ← Image | Depth | Depth-informed | ProxCAM
    │
    └── selected_samples/
        ├── images/                              ← Copy of selected pool images
        ├── labels/
        └── selection_log_iter_N.csv            ← filename + hybrid_score + per-method weights
```

### Key CSV columns — `training_summary.csv`

| Column | Description |
|---|---|
| `iteration` | AL round number (0–19) |
| `train_size` | Number of labelled training images |
| `pool_size` | Remaining unlabelled images |
| `test_accuracy` | Pixel accuracy on test set |
| `test_miou` | Mean IoU across all 9 classes |
| `val_miou` | Mean IoU on validation set |
| `epochs_trained` | Actual epochs run (may be < max due to early stopping) |
| `test_iou_<ClassName>` | Per-class IoU for each of the 9 classes |
| `avg_entropy_weight` | MLP-predicted weight for entropy method |
| `avg_unet_diversity_weight` | MLP-predicted weight for UNet diversity |
| `avg_dinov2_diversity_weight` | MLP-predicted weight for DINOv2 diversity |

---

## Resuming an Interrupted Run

The pipeline saves a full checkpoint after every iteration. If training is interrupted, simply re-run the same command:

```bash
python AAMEAL_Final_Code_with_Biplot.py
```

The script automatically detects `checkpoint.pth` in `RESULTS_DIR` and resumes from the next iteration. You will see:

```
✓ Resuming from iteration N
✓ Loaded N previous metrics
✓ MLP meta-learner restored from checkpoint
```

> If the checkpoint exists but the model `.pth` file for that iteration is missing, the script searches backwards to find the most recent valid model and resumes from there.

---


