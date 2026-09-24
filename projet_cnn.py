import os
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
import seaborn as sns

#-------------------------------------------------
# Configuration du device (GPU sidisponible)
#-------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")