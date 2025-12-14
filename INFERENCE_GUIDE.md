# OPT-RSVG Inference Guide

## Overview
This codebase implements **LPVA (Language-Guided Progressive Attention)** for Visual Grounding in Remote Sensing Images. The model takes an image and a text query as input and predicts a bounding box (normalized coordinates in xywh format).

## Architecture Summary

The LPVA model consists of five main components:

1. **Linguistic Backbone (BERT)**: Extracts features from text queries
2. **Progressive Attention Module**: Generates dynamic weights for visual backbone
3. **Visual Backbone (DETR-based ResNet)**: Extracts visual features from images
4. **Multi-Level Feature Enhancement Decoder**: Aggregates visual contextual information
5. **Localization Module**: Predicts the bounding box coordinates

## Inference Pipeline

### 1. Model Forward Pass Flow

```
Input: (img_data: NestedTensor, text_data: NestedTensor)
  ↓
1. Text Processing:
   - BERT encodes text → text_src [B, L, 768]
   - Extract CLS token → texts [B, 768]
   - Project to 256-dim → texts [B, 256]
  ↓
2. Visual Processing:
   - ResNet backbone extracts features (with language-guided attention)
   - DETR transformer processes features
   - Output: visu_mask, visu_src [N, B, C]
  ↓
3. Progressive Attention:
   - mhead1: visu_src × text_last → fv1
   - mhead2: visu_src × text_last → fc
   - mhead3: (fc + visu_src) × visu_src → fv2
   - Combined: visu_src = fv1 + fv2
  ↓
4. Vision-Language Fusion:
   - Concatenate: [reg_token, text_last, visu_src]
   - Vision-Language Transformer processes the concatenated features
  ↓
5. Bounding Box Prediction:
   - Extract regression token output
   - MLP head → pred_box [B, 4] (normalized xywh, sigmoid applied)
  ↓
Output: pred_box [B, 4] - normalized bounding box coordinates
```

### 2. Input Format

#### Image Input (`img_data`)
- **Type**: `NestedTensor` object
- **Structure**:
  - `img_data.tensors`: `[batch_size, 3, 640, 640]` - normalized image tensor
  - `img_data.mask`: `[batch_size, 640, 640]` - boolean mask (True for padding)
- **Preprocessing**:
  - Resize to 640×640 (maintaining aspect ratio, padded)
  - Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - Mask indicates padded regions (True = padding, False = valid pixels)

#### Text Input (`text_data`)
- **Type**: `NestedTensor` object
- **Structure**:
  - `text_data.tensors`: `[batch_size, max_query_len]` - BERT token IDs
  - `text_data.mask`: `[batch_size, max_query_len]` - attention mask (1 = valid, 0 = padding)
- **Preprocessing**:
  - Tokenize with BERT tokenizer
  - Add [CLS] and [SEP] tokens
  - Pad/truncate to `max_query_len` (default: 40)
  - Convert to token IDs

### 3. Output Format

- **Shape**: `[batch_size, 4]`
- **Format**: Normalized bounding box coordinates `[x_center, y_center, width, height]`
- **Range**: All values in [0, 1] (sigmoid applied)
- **Coordinate System**: 
  - (0, 0) is top-left corner
  - Values are relative to image size (640×640)
  - To convert to pixel coordinates: multiply by 640

## How to Run Inference

### Method 1: Using `eval.py` (Recommended)

```bash
python eval.py \
    --eval_model ./checkpoint/checkpoint.pth \
    --data_root ./ln_data/ \
    --split_root data \
    --dataset opt_rsvg \
    --eval_set test \
    --batch_size 16 \
    --device cuda \
    --output_dir ./outputs/ \
    --imsize 640 \
    --max_query_len 40 \
    --bert_model bert-base-uncased \
    --detr_model ./pretrained/detr-r50-e632da11.pth
```

**Key Arguments**:
- `--eval_model`: Path to trained checkpoint
- `--eval_set`: Dataset split to evaluate ('test', 'val', 'train')
- `--data_root`: Root directory containing dataset images
- `--split_root`: Directory containing split files (train.txt, val.txt, test.txt)
- `--dataset`: Dataset name ('opt_rsvg' or 'rsvgd')

### Method 2: Custom Inference Script

```python
import torch
from PIL import Image
from models import build_model
from datasets import build_dataset
from utils.misc import NestedTensor
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torchvision.transforms as T
from datasets.transforms import Compose, ToTensor, NormalizeAndPad

# 1. Load model
args = # ... your args object ...
model = build_model(args)
checkpoint = torch.load('path/to/checkpoint.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
model.to('cuda')

# 2. Prepare image
img = Image.open('path/to/image.jpg').convert('RGB')
transform = Compose([
    T.RandomResize([640, 640]),
    ToTensor(),
    NormalizeAndPad(size=640, aug_translate=False)
])
input_dict = transform({'img': img, 'box': torch.tensor([0, 0, 0, 0]), 'text': ''})
img_tensor = input_dict['img'].unsqueeze(0)  # [1, 3, 640, 640]
img_mask = input_dict['mask'].unsqueeze(0)   # [1, 640, 640]
img_data = NestedTensor(img_tensor, img_mask == 255)

# 3. Prepare text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
query = "a red car in the parking lot"
query = query.lower()
# Tokenize and convert to format expected by model
# ... (see data_loader.py for full tokenization logic)
text_ids = torch.tensor([token_ids]).to('cuda')  # [1, max_len]
text_mask = torch.tensor([attention_mask]).to('cuda')  # [1, max_len]
text_data = NestedTensor(text_ids, text_mask)

# 4. Run inference
with torch.no_grad():
    img_data = img_data.to('cuda')
    text_data = text_data.to('cuda')
    pred_box = model(img_data, text_data)  # [1, 4]

# 5. Convert to pixel coordinates
pred_box = pred_box[0].cpu()  # [4]
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

## Key Components

### Model Building (`models/__init__.py`)
```python
from models import build_model
model = build_model(args)  # Returns LPVA model
```

### Data Loading (`datasets/__init__.py`)
```python
from datasets import build_dataset
dataset = build_dataset('test', args)  # Returns LPVADataset
```

### Evaluation (`engine.py`)
- `evaluate()`: Full test set evaluation with metrics
- `validate()`: Validation set evaluation
- Returns: accuracy@0.5, accuracy@0.6, accuracy@0.7, accuracy@0.8, accuracy@0.9, meanIoU, cumuIoU

## Model Checkpoints

The model expects:
- **DETR pretrained weights**: `--detr_model` (default: `./pretrained/detr-r50-e632da11.pth`)
- **BERT model**: Loaded from `../bert/` or `bert-base-uncased`
- **Trained LPVA weights**: Loaded via `--eval_model` or `--resume`

## Important Notes

1. **Image Size**: Fixed at 640×640 pixels
2. **Text Length**: Maximum 40 tokens (including [CLS] and [SEP])
3. **Coordinate Format**: Output is normalized xywh (center coordinates + width/height)
4. **Device**: Model should be on same device as inputs
5. **Evaluation Mode**: Always use `model.eval()` for inference
6. **Batch Processing**: Model supports batched inference

## Evaluation Metrics

The evaluation computes:
- **Accuracy@0.5**: IoU ≥ 0.5
- **Accuracy@0.6**: IoU ≥ 0.6
- **Accuracy@0.7**: IoU ≥ 0.7
- **Accuracy@0.8**: IoU ≥ 0.8
- **Accuracy@0.9**: IoU ≥ 0.9
- **meanIoU**: Average IoU across all samples
- **cumuIoU**: Cumulative IoU (intersection area / union area)

## File Structure for Inference

```
OPT-RSVG/
├── eval.py                 # Main evaluation script
├── engine.py               # Evaluation functions
├── models/
│   ├── LPVA.py            # Main model architecture
│   ├── vl_transformer.py  # Vision-language transformer
│   └── ...
├── datasets/
│   ├── data_loader.py     # Dataset and data loading
│   └── transforms.py      # Image preprocessing
└── utils/
    ├── eval_utils.py      # Evaluation metrics
    └── misc.py            # Utilities (NestedTensor, collate_fn)
```

## Troubleshooting

1. **CUDA out of memory**: Reduce `--batch_size`
2. **Missing checkpoint**: Ensure checkpoint path is correct
3. **BERT model not found**: Check `--bert_model` path or download BERT
4. **Dataset not found**: Verify `--data_root` and `--split_root` paths
5. **Shape mismatches**: Ensure image size is 640×640 and text length matches `max_query_len`

