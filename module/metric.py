import torch
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max

from torchdrug.core import Registry as R
from sklearn.metrics import f1_score

@R.register("metrics.macrof1")
def macro_f1(pred, target):
    pred = pred.argmax(dim=-1)
    score = f1_score(target.cpu().data, pred.cpu(), average='macro', zero_division=0)
    return torch.tensor(score, dtype=torch.float32, device=pred.device)
    