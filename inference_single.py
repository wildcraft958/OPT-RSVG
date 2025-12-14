"""
Simple inference script for single image-text pair
Usage: python inference_single.py --image path/to/image.jpg --query "text query" --checkpoint path/to/checkpoint.pth
"""
import argparse
import torch
from PIL import Image
import numpy as np
from pathlib import Path

from models import build_model
from utils.misc import NestedTensor
from pytorch_pretrained_bert.tokenization import BertTokenizer
from datasets.transforms import Compose, ToTensor, NormalizeAndPad
import torchvision.transforms as T


def tokenize_query(query, tokenizer, max_len=40):
    """Tokenize a text query for BERT input"""
    query = query.lower().strip()
    
    # Simple tokenization (for full implementation, see data_loader.py)
    tokens = tokenizer.tokenize(query)
    
    # Add [CLS] and [SEP]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    # Truncate if too long
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    # Convert to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    
    # Pad to max_len
    while len(input_ids) < max_len:
        input_ids.append(0)
        input_mask.append(0)
    
    return input_ids, input_mask


def preprocess_image(image_path, imsize=640):
    """Preprocess image for model input"""
    img = Image.open(image_path).convert('RGB')
    
    transform = Compose([
        T.RandomResize([imsize, imsize]),
        ToTensor(),
        NormalizeAndPad(size=imsize, aug_translate=False)
    ])
    
    input_dict = transform({
        'img': img,
        'box': torch.tensor([0, 0, 0, 0]),  # Dummy box, not used for inference
        'text': ''
    })
    
    img_tensor = input_dict['img']  # [3, 640, 640]
    img_mask = input_dict['mask']   # [640, 640]
    
    return img_tensor, img_mask


def xywh_to_xyxy(box, img_size=640):
    """Convert center+wh format to x1y1x2y2 format"""
    x_center, y_center, width, height = box
    
    # Denormalize
    x_center *= img_size
    y_center *= img_size
    width *= img_size
    height *= img_size
    
    # Convert to x1y1x2y2
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [x1, y1, x2, y2]


def get_args_parser():
    parser = argparse.ArgumentParser('Single image inference for OPT-RSVG')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='LPVA')
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--imsize', default=640, type=int)
    parser.add_argument('--max_query_len', default=40, type=int)
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--emb_size', default=512, type=int)
    parser.add_argument('--vl_dropout', default=0.1, type=float)
    parser.add_argument('--vl_nheads', default=8, type=int)
    parser.add_argument('--vl_hidden_dim', default=256, type=int)
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int)
    parser.add_argument('--vl_enc_layers', default=6, type=int)
    parser.add_argument('--position_embedding', default='sine', type=str)
    
    # Input/Output
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--query', type=str, required=True, help='Text query')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str)
    parser.add_argument('--detr_model', default='./pretrained/detr-r50-e632da11.pth', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output', type=str, default=None, help='Path to save visualization (optional)')
    
    return parser


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Build model
    print('Building model...')
    model = build_model(args)
    model.to(device)
    model.eval()
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    print('Checkpoint loaded successfully!')
    
    # Preprocess image
    print(f'Loading image from {args.image}...')
    img_tensor, img_mask = preprocess_image(args.image, args.imsize)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, 640, 640]
    img_mask = img_mask.unsqueeze(0).to(device)       # [1, 640, 640]
    img_data = NestedTensor(img_tensor, img_mask == 255)
    
    # Preprocess text
    print(f'Tokenizing query: "{args.query}"...')
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    input_ids, input_mask = tokenize_query(args.query, tokenizer, args.max_query_len)
    text_ids = torch.tensor([input_ids]).to(device)      # [1, max_len]
    text_mask = torch.tensor([input_mask]).to(device)    # [1, max_len]
    text_data = NestedTensor(text_ids, text_mask)
    
    # Run inference
    print('Running inference...')
    with torch.no_grad():
        pred_box = model(img_data, text_data)  # [1, 4]
    
    pred_box = pred_box[0].cpu().numpy()  # [4] - normalized xywh
    
    # Convert to pixel coordinates
    bbox_xyxy = xywh_to_xyxy(pred_box, args.imsize)
    
    print('\n' + '='*50)
    print('RESULTS:')
    print('='*50)
    print(f'Normalized coordinates (xywh): [{pred_box[0]:.4f}, {pred_box[1]:.4f}, {pred_box[2]:.4f}, {pred_box[3]:.4f}]')
    print(f'Pixel coordinates (x1, y1, x2, y2): [{bbox_xyxy[0]:.1f}, {bbox_xyxy[1]:.1f}, {bbox_xyxy[2]:.1f}, {bbox_xyxy[3]:.1f}]')
    print('='*50)
    
    # Optional: Save visualization
    if args.output:
        try:
            from PIL import ImageDraw
            img = Image.open(args.image).convert('RGB')
            draw = ImageDraw.Draw(img)
            x1, y1, x2, y2 = bbox_xyxy
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            img.save(args.output)
            print(f'\nVisualization saved to {args.output}')
        except Exception as e:
            print(f'Warning: Could not save visualization: {e}')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

