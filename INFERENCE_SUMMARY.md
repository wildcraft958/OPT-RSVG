# OPT-RSVG Inference Analysis Summary

## Executive Summary

This codebase implements **LPVA (Language-Guided Progressive Attention)** for visual grounding in remote sensing images. The model takes an image and a natural language query as input and predicts a bounding box that localizes the object described in the query.

## Architecture Overview

### Model Components

1. **Linguistic Backbone (BERT)**
   - Location: `models/language_model/bert.py`
   - Purpose: Encodes text queries into semantic features
   - Output: Text features [B, L, 768] where L is sequence length

2. **Visual Backbone (DETR-based ResNet)**
   - Location: `models/visual_model/detr.py`, `models/visual_model/backbone.py`
   - Purpose: Extracts visual features from images with language-guided attention
   - Output: Visual features [N, B, C] where N is number of visual tokens

3. **Progressive Attention Module**
   - Location: `models/LPVA.py` (mhead1, mhead2, mhead3)
   - Purpose: Multi-level feature enhancement using cross-attention
   - Process: 
     - mhead1: Visual × Text → fv1
     - mhead2: Visual × Text → fc
     - mhead3: (fc + Visual) × Visual → fv2
     - Final: fv1 + fv2

4. **Vision-Language Transformer**
   - Location: `models/vl_transformer.py`
   - Purpose: Fuses visual and textual features
   - Input: Concatenated [reg_token, text_features, visual_features]

5. **Localization Head**
   - Location: `models/LPVA.py` (bbox_embed)
   - Purpose: Predicts bounding box coordinates
   - Output: Normalized [x_center, y_center, width, height]

## Inference Flow

```
Input Image (PIL Image) 
  → Resize to 640×640 + Normalize + Pad
  → NestedTensor(img_tensor, img_mask)
  ↓
Text Query (string)
  → BERT Tokenization
  → NestedTensor(text_ids, text_mask)
  ↓
Model Forward Pass:
  1. BERT encoding → text features
  2. Visual backbone (with language-guided attention) → visual features
  3. Progressive attention → enhanced visual features
  4. Vision-language fusion → fused features
  5. Bounding box prediction → [x, y, w, h]
  ↓
Output: Normalized bounding box coordinates [4]
```

## Key Files for Inference

### Entry Points
- **`eval.py`**: Full dataset evaluation script
- **`inference_single.py`**: Single image inference script (newly created)

### Core Components
- **`models/LPVA.py`**: Main model architecture
- **`models/__init__.py`**: Model builder function
- **`engine.py`**: Evaluation functions (`evaluate()`, `validate()`)
- **`datasets/__init__.py`**: Dataset builder
- **`datasets/data_loader.py`**: Data loading and preprocessing
- **`datasets/transforms.py`**: Image transformations
- **`utils/misc.py`**: Utilities (NestedTensor, collate_fn)
- **`utils/eval_utils.py`**: Evaluation metrics

## Input/Output Specifications

### Input Format

**Image:**
- Type: PIL Image (RGB)
- Preprocessing:
  - Resize to 640×640 (maintain aspect ratio, pad with zeros)
  - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - Convert to tensor: [3, 640, 640]
  - Create mask: [640, 640] (True for padding, False for valid pixels)
- Final format: `NestedTensor(tensor, mask)`

**Text:**
- Type: String (natural language query)
- Preprocessing:
  - Lowercase
  - BERT tokenization (add [CLS] and [SEP])
  - Pad/truncate to max_query_len (default: 40)
  - Convert to token IDs
- Final format: `NestedTensor(token_ids, attention_mask)`

### Output Format

**Bounding Box:**
- Shape: [batch_size, 4]
- Format: Normalized coordinates [x_center, y_center, width, height]
- Range: [0, 1] (sigmoid applied)
- Coordinate system:
  - (0, 0) = top-left corner
  - Values relative to 640×640 image size
  - To convert to pixels: multiply by 640

## How to Perform Inference

### Method 1: Using eval.py (Full Dataset)

```bash
python eval.py \
    --eval_model ./checkpoint/checkpoint.pth \
    --data_root ./ln_data/ \
    --split_root data \
    --dataset opt_rsvg \
    --eval_set test \
    --batch_size 16 \
    --device cuda
```

### Method 2: Using inference_single.py (Single Image)

```bash
python inference_single.py \
    --image path/to/image.jpg \
    --query "a red car in the parking lot" \
    --checkpoint path/to/checkpoint.pth \
    --device cuda \
    --output result.jpg  # Optional: save visualization
```

### Method 3: Programmatic Usage

```python
import torch
from models import build_model
from utils.misc import NestedTensor

# Build and load model
model = build_model(args)
checkpoint = torch.load('checkpoint.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
model.to('cuda')

# Prepare inputs (img_data and text_data as NestedTensor)
with torch.no_grad():
    pred_box = model(img_data, text_data)  # [B, 4]
```

## Model Configuration

### Default Hyperparameters

- **Image size**: 640×640
- **Max query length**: 40 tokens
- **Backbone**: ResNet-50
- **Hidden dimension**: 256
- **BERT encoder layers**: 12
- **DETR encoder layers**: 6
- **Vision-language encoder layers**: 6
- **Number of attention heads**: 8
- **Feedforward dimension**: 2048

### Required Checkpoints

1. **DETR pretrained weights**: `./pretrained/detr-r50-e632da11.pth`
2. **BERT model**: `bert-base-uncased` (or local path `../bert/`)
3. **Trained LPVA weights**: Loaded via `--eval_model` or `--resume`

## Evaluation Metrics

The evaluation computes multiple metrics:

- **Accuracy@0.5**: Percentage of predictions with IoU ≥ 0.5
- **Accuracy@0.6**: Percentage of predictions with IoU ≥ 0.6
- **Accuracy@0.7**: Percentage of predictions with IoU ≥ 0.7
- **Accuracy@0.8**: Percentage of predictions with IoU ≥ 0.8
- **Accuracy@0.9**: Percentage of predictions with IoU ≥ 0.9
- **meanIoU**: Average Intersection over Union
- **cumuIoU**: Cumulative IoU (total intersection / total union)

## Data Format

### Dataset Structure (OPT-RSVG)

```
data/
├── opt_rsvg/
│   ├── train.txt      # List of training sample indices
│   ├── val.txt         # List of validation sample indices
│   └── test.txt        # List of test sample indices
└── opt_rsvg/
    ├── JPEGImages/     # Image files
    └── Annotations/    # XML annotation files
```

### Annotation Format (XML)

Each XML file contains:
- `<filename>`: Image filename
- `<object>`:
  - `<name>`: Object class
  - `<bndbox>`: Bounding box [x1, y1, x2, y2]
  - `<query>`: Text description

## Important Implementation Details

### NestedTensor

The codebase uses a custom `NestedTensor` class to handle variable-sized inputs:
- Contains `tensors` (actual data) and `mask` (padding mask)
- Used for both images and text
- Automatically handles device placement

### Progressive Attention

The progressive attention mechanism:
1. First applies cross-attention between visual and text features
2. Adds residual connection
3. Applies self-attention on enhanced visual features
4. Combines both attention outputs

### Coordinate Conversion

The model outputs normalized xywh format. To convert to pixel coordinates:

```python
# Denormalize
x_center *= 640
y_center *= 640
width *= 640
height *= 640

# Convert to x1y1x2y2
x1 = x_center - width / 2
y1 = y_center - height / 2
x2 = x_center + width / 2
y2 = y_center + height / 2
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Solution: Reduce `--batch_size`

2. **Missing checkpoint**
   - Check: `--eval_model` path is correct
   - Check: Checkpoint contains 'model' key or is direct state_dict

3. **BERT model not found**
   - Check: `--bert_model` path or download BERT
   - Default: Looks for `../bert/` or downloads `bert-base-uncased`

4. **Shape mismatches**
   - Ensure: Image size is 640×640
   - Ensure: Text length matches `max_query_len` (40)

5. **Dataset not found**
   - Verify: `--data_root` contains dataset images
   - Verify: `--split_root` contains split files (train.txt, val.txt, test.txt)

## Performance

According to the README, the model achieves:
- **OPT-RSVG test set**: 78.03% Pr@0.5, 66.20% meanIoU, 76.30% cmuIoU
- **DIOR-RSVG test set**: 82.27% Pr@0.5, 72.35% meanIoU, 85.11% cmuIoU

## Next Steps

1. **For evaluation**: Use `eval.py` with appropriate dataset paths
2. **For single inference**: Use `inference_single.py` 
3. **For custom integration**: Follow the model forward pass in `models/LPVA.py`
4. **For batch processing**: Use the DataLoader from `datasets/__init__.py`

## Additional Resources

- **Inference Guide**: See `INFERENCE_GUIDE.md` for detailed documentation
- **Model Architecture**: See `models/LPVA.py` for implementation details
- **Data Loading**: See `datasets/data_loader.py` for preprocessing logic

