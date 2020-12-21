import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import glob
import shutil
import pandas as pd
import re
import torch.nn.functional as F
import pdb
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import configparser
from .Dataset.dataset import data_set, get_dataloaders
from torch.utils.data import DataLoader



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
torch.cuda.manual_seed_all(3)
np.random.seed(2)


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')


    plot_dl = config.getboolean('Params', 'plot_dataloaders')

    # ----- Load data ----- #
    train_dataloader, val_dataloader, _ = get_dataloaders(config=config)

    # ----- model ----- #


    # ----- losses ----- #


if __name__ == '__main__':
    label_list = io.loadmat('./clothes_data/label_list.mat')
    num_of_classes = label_list['label_list'].shape[1]
    softmax = torch.nn.Softmax(dim=1)



