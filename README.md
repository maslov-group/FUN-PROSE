# Fungi_Gene_Expression

The processed data and test outputs can be found here: https://drive.google.com/drive/folders/19K5DxFVpjozpd1rKnBvZZzDmnPp1-Ldw?usp=sharing

Required versions: Ray v1.8.0, PyTorch v1.9.1, and CUDA v11.5

```
import torch
from classifier.py import *
model = torch.load(PATH)
model.eval()
```
