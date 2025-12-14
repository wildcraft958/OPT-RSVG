# OPT-RSVG Inference Quick Reference

## Quick Start

### Single Image Inference
```bash
python inference_single.py \
    --image image.jpg \
    --query "text description" \
    --checkpoint checkpoint.pth
```

### Full Dataset Evaluation
```bash
python eval.py \
    --eval_model checkpoint.pth \
    --data_root ./ln_data/ \
    --dataset opt_rsvg \
    --eval_set test
```

## Model Input/Output

| Component | Format | Shape | Notes |
|-----------|--------|-------|-------|
| **Image Input** | NestedTensor | `[B, 3, 640, 640]` | Normalized, padded to 640×640 |
| **Text Input** | NestedTensor | `[B, 40]` | BERT token IDs, max 40 tokens |
| **Output** | Tensor | `[B, 4]` | Normalized [x_center, y_center, width, height] |

## Model Architecture

```
Image → ResNet-50 → DETR Encoder → Visual Features
Text → BERT → Text Features
↓
Progressive Attention (3 stages)
↓
Vision-Language Transformer
↓
Bounding Box Head → [x, y, w, h]
```

## Key Code Locations

| Component | File | Function/Class |
|-----------|------|----------------|
| Main Model | `models/LPVA.py` | `LPVA.forward()` |
| Model Builder | `models/__init__.py` | `build_model()` |
| Evaluation | `engine.py` | `evaluate()`, `validate()` |
| Data Loading | `datasets/__init__.py` | `build_dataset()` |
| Image Transform | `datasets/transforms.py` | `NormalizeAndPad` |
| Text Tokenization | `datasets/data_loader.py` | `convert_examples_to_features()` |

## Default Configuration

```python
imsize = 640
max_query_len = 40
backbone = 'resnet50'
hidden_dim = 256
bert_enc_num = 12
detr_enc_num = 6
vl_enc_layers = 6
nheads = 8
dim_feedforward = 2048
```

## Coordinate Conversion

```python
# Model output: normalized [x_center, y_center, width, height]
# Convert to pixel coordinates (x1, y1, x2, y2):

x_center, y_center, width, height = pred_box
x_center *= 640
y_center *= 640
width *= 640
height *= 640

x1 = x_center - width / 2
y1 = y_center - height / 2
x2 = x_center + width / 2
y2 = y_center + height / 2
```

## Evaluation Metrics

- **Pr@0.5**: Accuracy with IoU ≥ 0.5
- **Pr@0.6-0.9**: Accuracy at higher IoU thresholds
- **meanIoU**: Average Intersection over Union
- **cumuIoU**: Cumulative IoU (total intersection / total union)

## Common Commands

```bash
# Evaluation with custom paths
python eval.py --eval_model model.pth --data_root ./data --dataset opt_rsvg

# Single inference with visualization
python inference_single.py --image img.jpg --query "query" --checkpoint model.pth --output result.jpg

# Training (for reference)
python train.py --data_root ./data --output_dir ./output
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `--batch_size` |
| Missing checkpoint | Check `--eval_model` path |
| BERT not found | Set `--bert_model` or download BERT |
| Shape mismatch | Ensure image=640×640, text=max_query_len=40 |

## File Structure

```
OPT-RSVG/
├── eval.py              # Full evaluation script
├── inference_single.py  # Single image inference
├── train.py            # Training script
├── engine.py           # Evaluation functions
├── models/
│   ├── LPVA.py         # Main model
│   └── ...
├── datasets/
│   ├── data_loader.py  # Data loading
│   └── transforms.py   # Preprocessing
└── utils/
    ├── eval_utils.py   # Metrics
    └── misc.py         # Utilities
```

