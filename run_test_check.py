import json
import numpy as np
from PIL import Image
import onnxruntime as ort

print("=" * 65)
print("TEST CHECK ON TEST LEAF (check_image.jpg)")
print("=" * 65)

# 1. Load model info
with open("model_info.json", "r") as f:
    info = json.load(f)

classes = info["classes"]
input_size = info["input_size"]
mean = np.array(info["mean"], dtype=np.float32)
std = np.array(info["std"], dtype=np.float32)

print(f"Model input dimensions : {input_size} x {input_size}")
print(f"Target classes         : {classes}")
print(f"Channel mean           : {mean.tolist()}")
print(f"Channel std            : {std.tolist()}")

# 2. Load test leaf image
img = Image.open("check_image.jpg").convert("RGB")
w, h = img.size
print(f"\nTest image loaded      : check_image.jpg ({w}x{h} px)")

# 3. Crop central square and resize
side = min(w, h)
left = (w - side) // 2
top = (h - side) // 2
img_cropped = img.crop((left, top, left + side, top + side))
img_resized = img_cropped.resize((input_size, input_size), Image.BILINEAR)

# 4. Standardize
arr = np.array(img_resized, dtype=np.float32) / 255.0
arr = (arr - mean) / std
tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, :, :, :]
print(f"Input tensor shape     : {tensor.shape}")

# 5. Run ONNX Inference
session = ort.InferenceSession("leaf_model.onnx")
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name
logits = session.run([out_name], {inp_name: tensor})[0][0]

print("\nRaw Logits from leaf_model.onnx:")
for cls, logit in zip(classes, logits):
    print(f"  {cls:10s}: {logit:10.5f}")

# 6. Apply Softmax
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)

# 7. Compare with check_expected.json
with open("check_expected.json", "r") as f:
    expected = json.load(f)

print("\n" + "-" * 65)
print(f"{'Class':10s} | {'Calculated Prob':16s} | {'Expected Prob':16s} | {'Abs Diff':10s} | Status")
print("-" * 65)

all_pass = True
max_diff = 0.0

for cls, prob in zip(classes, probs):
    exp = expected[cls]
    diff = abs(prob - exp)
    if diff > max_diff:
        max_diff = diff
    passed = diff < 1e-4
    if not passed:
        all_pass = False
    status = "PASS" if passed else "FAIL"
    print(f"{cls:10s} | {prob:16.8f} | {exp:16.8f} | {diff:10.2e} | {status}")

print("-" * 65)
winner_idx = np.argmax(probs)
winner = classes[winner_idx]
print(f"\nTop Predicted Class  : {winner} ({probs[winner_idx] * 100:.2f}%)")
print(f"Max Absolute Error   : {max_diff:.4e}")
print(f"Check Verdict        : {'PASSED (Exact Match)' if all_pass else 'FAILED'}")
