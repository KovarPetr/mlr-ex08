# Leaf Classifier Live

Live web application that classifies leaves in real time from the phone's rear camera using ONNX Runtime Web.

🔗 **Live Web App:** [https://kovarpetr.github.io/mlr-ex08/](https://kovarpetr.github.io/mlr-ex08/)

---

## Overview

This project runs deep learning inference directly on-device in the browser using [ONNX Runtime Web](https://github.com/microsoft/onnxruntime). The model (`leaf_model.onnx`) classifies leaves into four distinct species:

- 🍏 **Apple** (`apple`)
- 🍒 **Cherry** (`cherry`)
- 🌰 **Chestnut** (`chestnut`)
- 🍁 **Maple** (`maple`)

---

## Technical Specifications

The inference pipeline follows `model_info.json` exactly:

- **Input size:** `96 × 96` pixels
- **Input shape:** `1 × 3 × 96 × 96` (NCHW channel-first format)
- **Central Square Cropping:** Extracts the maximum centered square from each frame / image:
  $$\text{side} = \min(W, H), \quad x = \frac{W - \text{side}}{2}, \quad y = \frac{H - \text{side}}{2}$$
- **Resize:** Resized to `96 × 96`
- **Pixel Scaling:** RGB channels scaled to range $[0, 1]$
- **Channel Standardization:**
  - Mean: $[0.485, 0.456, 0.406]$
  - Std: $[0.229, 0.224, 0.225]$
  $$\text{normalized} = \frac{\text{pixel} - \text{mean}}{\text{std}}$$
- **Classification Output:** Logits converted to probabilities via softmax:
  $$P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

---

## Features

- **Live Rear Camera Classification:** Uses `navigator.mediaDevices.getUserMedia` with `{ facingMode: { ideal: 'environment' } }`.
- **Safari on iPhone Compatible:** Configured with `playsinline`, `webkit-playsinline`, `autoplay`, and `muted` attributes, alongside single-threaded WebAssembly execution provider.
- **Real-Time Probability Bars:** Live animated percentage bars and top-prediction highlight.
- **Photo Upload Fallback:** Upload photos directly from storage or camera roll.
- **Self-Check Verification:** Built-in test that evaluates `check_image.jpg` against `check_expected.json`.
- **In-Browser / Local QR Code:** Clearly labelled QR code rendered locally and dynamically in the browser that points directly to the live deployment.

---

## Accuracy Verification (`check_image.jpg`)

| Class | Expected Probability | Calculated Probability | Difference |
| :--- | :--- | :--- | :--- |
| **Apple** | `0.457831` | `0.457827` | $4.35 \times 10^{-6}$ |
| **Cherry** | `0.520016` | `0.520020` | $3.28 \times 10^{-6}$ |
| **Chestnut** | `0.001382` | `0.001382` | $1.65 \times 10^{-8}$ |
| **Maple** | `0.020770` | `0.020771` | $1.09 \times 10^{-6}$ |

---

## Repository Structure

- `index.html` — Complete standalone application (HTML, CSS, plain JavaScript).
- `leaf_model.onnx` — Trained ONNX model.
- `model_info.json` — Model parameters (input size, classes, mean, std).
- `check_image.jpg` — Reference test image.
- `check_expected.json` — Reference output probabilities.
- `qr_code.png` — Verified QR code pointing to live GitHub Pages URL.
- `README.md` — Project documentation and live link.
