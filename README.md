Fungi_Gene_Expression

The processed data and test outputs can be found here: https://drive.google.com/drive/folders/19K5DxFVpjozpd1rKnBvZZzDmnPp1-Ldw?usp=sharing

```
import torch
from classifier.py import *
model = torch.load(PATH)
model.eval()
```
