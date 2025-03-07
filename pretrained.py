import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from tqdm import tqdm
from librispeech import LibriSpeech
import numpy as np
import random

TRAIN_DATASET = "dev-clean"
VAL_DATASET = "dev-other"  
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"


