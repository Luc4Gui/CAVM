import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def custom_transform_pre(data):
    loadSize = 192
    ch = 128
    B, C, W, H, D = data.shape
    data = data[:, :, W//2-loadSize//2:W//2+loadSize//2, H//2-loadSize//2:H//2+loadSize//2, D//2-ch//2:D//2+ch//2]
    return data
