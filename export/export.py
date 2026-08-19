"""Export the default models to ONNX.

Downloads the DRAW 2 Yu-Gi-Oh! detector (AGPL-3.0) and DINOv2-small
(Apache-2.0) from Hugging Face, then writes detector.onnx and embedder.onnx
into the output directory (default: current directory).

Usage: python export.py [out_dir]
"""
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel
from ultralytics import YOLO

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'

weights = hf_hub_download('HichTala/draw2', 'ygo_yolo.pt')
path = YOLO(weights).export(format='onnx', imgsz=640, device='cpu')
os.replace(path, os.path.join(OUT, 'detector.onnx'))
print('detector.onnx', os.path.getsize(os.path.join(OUT, 'detector.onnx')) // 1024, 'KB')


class Embedder(torch.nn.Module):
    """DINOv2-small with L2(concat(L2(cls), L2(mean(patches)))) pooling."""

    def __init__(self):
        super().__init__()
        self.m = AutoModel.from_pretrained('facebook/dinov2-small')

    def forward(self, x):
        out = self.m(pixel_values=x).last_hidden_state
        cls = torch.nn.functional.normalize(out[:, 0], dim=-1)
        pat = torch.nn.functional.normalize(out[:, 1:].mean(1), dim=-1)
        return torch.nn.functional.normalize(torch.cat([cls, pat], dim=-1), dim=-1)


model = Embedder().eval()
torch.onnx.export(
    model, torch.zeros(1, 3, 224, 224), os.path.join(OUT, 'embedder.onnx'),
    input_names=['pixel_values'], output_names=['emb'],
    dynamic_axes={'pixel_values': {0: 'b'}, 'emb': {0: 'b'}}, opset_version=17,
)
print('embedder.onnx', os.path.getsize(os.path.join(OUT, 'embedder.onnx')) // 1024, 'KB')
