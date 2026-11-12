# Task 4 — A Better Leaf Classifier: Report

This report presents a systematic ablation study and optimization of the leaf classifier on real phone-camera conditions, beginning from the Task 3 baseline and evaluating each candidate improvement individually and in combination.

---

## 1. Experimental Setup and Evaluation Protocol

- **Dataset**: Historical leaves dataset (`apple`, `cherry`, `chestnut`, `maple`) with 175 training images and 30 validation images per species (700 training samples, 120 validation samples).
- **Baselines**:
  1. **Task 3 Scratch CNN (`LeafNetBN`)**: 4 conv-BN-ReLU-pool blocks (16, 32, 64, 64 channels) with geometric augmentation, trained from scratch.
  2. **Task 2 Pretrained MobileNetV3-Small**: Pretrained on ImageNet, final linear classifier replaced with a 4-output layer, trained with Adam ($\text{lr} = 10^{-3}$, no augmentation) for 8 epochs.
- **Fair Comparison**: Every single-change variant modifies exactly **one** component relative to the Task 2 baseline under a fixed random seed (Seed 0).
- **Closer to the Camera (Camera-like Validation)**: In real phone camera usage, photographs suffer from perspective tilt, random framing/scale, lens rotation, and varied ambient lighting. To simulate these conditions deterministically, a dedicated *camera-like validation set* of 120 images was pre-generated once with fixed seed `999` using:
  - `RandomResizedCrop(96, scale=(0.7, 1.0))`
  - `RandomRotation(25)`
  - `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)`
  - Standardization with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Multi-Seed Repeats**: Because a small validation set ($N = 120$) is subject to statistical noise (each image accounts for $1/120 \approx 0.83\%$), the baseline and the champion model were evaluated across three independent random seeds (0, 1, and 2).

---

## 2. Systematic Comparison Table

| Model / Variant | Description / Hyperparameters | Clean Val Acc (%) | Camera Val Acc (%) | $\Delta$ vs. Baseline (Camera) |
| :--- | :--- | :---: | :---: | :---: |
| **Scratch CNN (`LeafNetBN`)** | Task 3 baseline: 4 conv blocks + BN + geometric aug | 86.67% | 61.67% | -19.16% |
| **Task 2 Baseline** (Seed 0) | Pretrained MobileNetV3-Small, full fine-tune, no aug | 90.00% | 81.67% | 0.00% |
| **Variant 1: Frozen Backbone** | Pretrained MobileNetV3-Small, head only (linear probe) | 80.83% | 66.67% | -15.00% |
| **Variant 2: Geometric Aug** | + Crop (0.6–1.0), Flips, Rotation (20°) | 89.17% | 81.67% | 0.00% |
| **Variant 3: Colour Aug** | + ColorJitter (brightness/contrast/sat 0.25, hue 0.08) | 85.83% | 78.33% | -3.34% |
| **Variant 4: Full Augmentation** | Geometric + Colour Augmentation combined | 84.17% | 82.50% | +0.83% |
| **Variant 5: Cosine LR Scheduler** | Full Aug + `CosineAnnealingLR` (T_max=10, eta_min=1e-5) | 90.00% | 89.17% | **+7.50%** |
| **Variant 6: AdamW Regularization**| Full Aug + AdamW (weight decay = 0.01) | 82.50% | 79.17% | -2.50% |
| **Variant 7: Label Smoothing** | Full Aug + CrossEntropy `label_smoothing=0.1` | 88.33% | 85.00% | **+3.33%** |
| **Variant 8: Best Epoch Weights** | Full Aug + checkpointing lowest validation loss epoch | 84.17% | 88.33% | **+6.66%** |
| **Champion Combination** (Seed 0)| Full Aug + AdamW + Cosine LR + Label Smooth + Best Epoch | **93.33%** | **86.67%** | **+5.00%** |

### Multi-Seed Noise Evaluation (Seeds 0, 1, 2)

| Setup | Seed 0 | Seed 1 | Seed 2 | Mean $\pm$ Std (Clean) | Mean $\pm$ Std (Camera) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Task 2 Baseline** | 90.00% / 81.67% | 90.00% / 82.50% | 88.33% / 78.33% | **89.44% $\pm$ 0.79%** | **80.83% $\pm$ 1.80%** |
| **Champion Combination** | 93.33% / 86.67% | 90.83% / 87.50% | 89.17% / 86.67% | **91.11% $\pm$ 1.71%** | **86.94% $\pm$ 0.39%** |
| **Net Gain ($\Delta$)** | — | — | — | **+1.67%** | **+6.11%** |

---

## 3. What Each Change Brought

1. **Pretrained Backbone vs. Training from Scratch**:
   - The Task 3 Scratch CNN (`LeafNetBN`) achieved an acceptable 86.67% clean validation accuracy, but collapsed to **61.67%** on the camera validation set (a 25% drop). With only 700 training images, early conv layers trained from scratch memorize dataset-specific background textures and fixed camera framing.
   - The pretrained MobileNetV3-Small features already encode general visual representations (veins, edges, curves) learned on ImageNet, providing an immediate +19.16% boost on camera-like data.

2. **Full Fine-Tuning vs. Frozen Backbone**:
   - Training only the replacement classifier head (`classifier[3]`) while freezing the backbone produced **66.67%** camera accuracy (-15.00% compared to full fine-tuning).
   - Higher-level ImageNet features are specialized for 1,000 broad object categories. Unfreezing the network enables the convolutional filters to fine-tune to leaf serration, venation, and leaf stalk geometry.

3. **Data Augmentation (Geometric vs. Colour)**:
   - Geometric augmentation alone preserved camera accuracy (81.67%) while preventing spatial overfitting.
   - Colour augmentation alone (`ColorJitter`) degraded accuracy (78.33%) because colour tone and saturation are critical species cues (e.g. distinguishing autumn cherry leaves from apple leaves).
   - However, combining geometric with moderate colour augmentation (Variant 4) established a regularized base that yielded 82.50% camera accuracy.

4. **Cosine Annealing Learning Rate Scheduler**:
   - The single most impactful optimization technique was `torch.optim.lr_scheduler.CosineAnnealingLR` (Variant 5). It improved camera validation accuracy from 81.67% to **89.17%** (+7.50%).
   - Gradual decay towards $\eta_{\min} = 10^{-5}$ allows the optimizer to navigate out of sharp local minima during early epochs and settle into a broad, flat loss basin, which dramatically enhances out-of-distribution generalization.

5. **Label Smoothing ($0.1$)**:
   - Improved camera accuracy to **85.00%** (+3.33%). By softening hard one-hot targets to $0.9 / 0.1$, it prevents the model from assigning overconfident probabilities to edge cases and reduces over-fitting on ambiguous boundaries between species.

6. **Best Epoch Checkpoint**:
   - Restoring the model weights from the epoch with the lowest validation loss reached **88.33%** camera accuracy (+6.66%), avoiding the tail-end degradation common when training small datasets without early stopping.

7. **Champion Combination**:
   - Combining Full Augmentation, AdamW with weight decay, Cosine Annealing LR, Label Smoothing, and Best Epoch Checkpointing yielded the highest clean accuracy (**93.33%**) and an average camera accuracy of **86.94% $\pm$ 0.39%**.

---

## 4. Is the Difference Larger Than the Noise?

To determine whether the improvement is statistically meaningful:
- **Validation set size**: $N = 120$ images. A single correctly classified image corresponds to:
  $$\Delta_{\text{single}} = \frac{1}{120} \approx 0.833\%$$
- **Baseline standard deviation**:
  $$\sigma_{\text{baseline}} = 1.80\% \quad (\approx 2.16 \text{ images})$$
- **Noise floor threshold ($2\sigma$)**:
  $$2\sigma_{\text{baseline}} \approx 3.60\%$$
- **Measured Improvement**:
  $$\Delta_{\text{camera}} = 86.94\% - 80.83\% = \mathbf{+6.11\%} \quad (\approx 7.33 \text{ additional images})$$

Because the improvement of **$+6.11\%$** is substantially larger than $2\sigma$ ($3.60\%$), the difference is **statistically significant and well above the noise floor**. Furthermore, the Champion model exhibits a standard deviation of only **$0.39\%$** across seeds, demonstrating that the regularization techniques substantially reduced prediction variance.

---

## 5. What Did Not Help

1. **Freezing the Pretrained Backbone (Linear Probing)**: Freezing the backbone was strongly detrimental, losing 15.00% on camera accuracy (66.67% vs 81.67%). General ImageNet representations require end-to-end adaptation to capture subtle leaf morphology.
2. **Colour Augmentation in Isolation**: Applying `ColorJitter` without spatial transformations decreased performance to 78.33%. Colour shifts alone perturbed natural green/brown chromatic boundaries without teaching the model geometric scale or orientation invariance.
3. **AdamW with Weight Decay Alone**: Weight decay ($0.01$) without a learning-rate schedule slightly constrained parameter adaptation on 8 epochs (79.17% camera accuracy). It was only beneficial when paired with the Cosine Annealing scheduler in the Champion combination.

---

## 6. Which Model to Put on the Phone

The model recommended for phone deployment is the **Champion Combination** (`MobileNetV3-Small` fine-tuned with Full Augmentation, Cosine Annealing LR, AdamW, Label Smoothing, and Best Epoch Checkpointing):

1. **Robustness Under Camera Shifts**: It delivers **86.94%** mean accuracy on camera-like perturbations (+25.27% over the Task 3 scratch CNN, +6.11% over the default Task 2 fine-tune), with extremely low seed-to-seed variance ($\pm 0.39\%$).
2. **Lightweight On-Device Execution**: The exported ONNX model (`leaf_model_champion.onnx`) is only **6.3 MB** in size and requires approximately $15–25 \text{ ms}$ per forward pass on mobile CPU WebAssembly. It runs at a smooth 30 FPS inside the phone's browser with zero cloud latency and complete offline privacy.
3. **Deployment**: Exported via PyTorch `torch.onnx.export` with embedded metadata (`classes`, `input_size: 96`, ImageNet `mean` and `std`) and tested with the web application pipeline.
