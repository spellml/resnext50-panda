"""
Initial model definition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from fastai.layers import AdaptiveConcatPool2d, Flatten
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

import os
import io
import zipfile


LABELS = '/mnt/prostate-cancer-grade-assessment/train.csv'
OUT_TRAIN = '/mnt/prostate-cancer-grade-assessment/train.zip'
OUT_MASKS = '/mnt/prostate-cancer-grade-assessment/masks.zip'
NUM_EPOCHS = 16
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.8


##############
# DATALOADER #
##############

def get_train_test_split():
    # NOTE(aleksey): the DataFrame contains 100 entries which are mysteriously absent from
    # the train set. Here we filter these out here.
    with zipfile.ZipFile(OUT_TRAIN) as zf_train:
        files = [item.filename for item in zf_train.infolist()]
        files = [f[:32] for f in files]
        files = set(files)

        df = pd.read_csv(LABELS).set_index('image_id')
        df = df.loc[files]
    df = df.reset_index()
    train_df = df.sample(len(df) // TRAIN_TEST_SPLIT)
    test_df = df.loc[df.index.difference(train_df.index)]
    return train_df, test_df


class PandaDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.image_zf = zipfile.ZipFile(OUT_TRAIN)
        self.mask_zf = zipfile.ZipFile(OUT_MASKS)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        X_id = self.df.iloc[idx, 0]
        # The target variable is the isup_score, which is a value 0 through 5.
        # The moodel predicts six values per class (a la softmax).
        y = torch.tensor(self.df.iloc[idx].isup_grade)
        tile_img_tensors, tile_mask_tensors = [], []

        for tile_id in range(0, 16):
            fd = self.image_zf.open(f"{X_id}_{tile_id}.png")
            tile_img_bytes = fd.read()
            fd.close()
            
            tile_img = Image.open(io.BytesIO(tile_img_bytes))
            tile_img_arr = np.array(tile_img)
            tile_img_arr.astype(np.float)
            
            fd = self.mask_zf.open(f"{X_id}_{tile_id}.png")
            tile_mask_bytes = fd.read()
            fd.close()
            
            tile_mask = Image.open(io.BytesIO(tile_mask_bytes))
            tile_mask_arr = np.array(tile_mask)
            
            tile_img_tensor = torch.tensor(tile_img_arr, dtype=torch.float).permute(2, 0, 1)
            tile_img_tensor = transforms.Normalize(
                mean=tile_img_tensor_means, std=tile_img_tensor_stds
            )(tile_img_tensor)

            tile_mask_tensor = torch.tensor(tile_mask_arr, dtype=torch.float)[np.newaxis, ...]
            tile_mask_tensor = transforms.Normalize(
                mean=(tile_mask_tensor_mean), std=(tile_mask_tensor_std)
            )(tile_mask_tensor)
            
            tile_img_tensors.append(tile_img_tensor)
            tile_mask_tensors.append(tile_mask_tensor)
        
        return torch.cat(tile_img_tensors, dim=1), torch.cat(tile_img_tensors, dim=1), y


#########
# MODEL #
#########

# NOTE(aleksey): these magic values are used by torchvision.transforms.Normalize.
def estimate_mean_and_std(zf_path, n_samples=1000):
    def calculate_mean_and_std(img_arr):
        return np.mean(img_arr, axis=(0, 1)), np.std(img_arr, axis=(0, 1))

    all_per_channel_means = []
    all_per_channel_stds = []
    
    with zipfile.ZipFile(zf_path) as zf:
        for img_info in np.random.choice(zf.infolist(), n_samples):
            img_fn = img_info.filename
            with zf.open(img_fn) as fp:
                img = Image.open(io.BytesIO(fp.read()))
                img_arr = np.array(img)
                per_channel_means, per_channel_stds = calculate_mean_and_std(img_arr)
                all_per_channel_means.append(per_channel_means)
                all_per_channel_stds.append(per_channel_stds)
    
    estimated_channelwide_means = np.stack(all_per_channel_means).mean(axis=0)
    estimated_channelwide_stds = np.stack(all_per_channel_stds).mean(axis=0)
    return estimated_channelwide_means, estimated_channelwide_stds

tile_img_tensor_means, tile_img_tensor_stds = estimate_mean_and_std(OUT_TRAIN)
tile_mask_tensor_mean, tile_mask_tensor_std = estimate_mean_and_std(OUT_MASKS)

# NOTE(aleksey): the model uses a Mish() module as part of its architecture. In the original
# Kaggle kernel this model is sourced from the mish_activation package. I could not determine
# the origin of this module, but inspecting the module in a Kaggle notebook (via import
# mish_activation; mish_activation??) reveals that the activation definition is a simple flat
# file. I've gone ahead and copied that to here.
class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)


class PandaModel(nn.Module):
    # NOTE(aleksey): n=6 because the value we are trying to predict is an ordinal categorical
    # with 6 possible values.
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            # TODO: replace these fastaia layers with torch ones
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(2 * nc, 512),
            Mish(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, n)
        )
        
    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x, 1).view(-1, shape[1], shape[2], shape[3])
        
        #x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = (x.view(-1, n, shape[1], shape[2], shape[3])
             .permute(0, 2, 1, 3, 4)
             .contiguous()
             .view(-1, shape[1], shape[2] * n, shape[3]))

        #x: bs x C x N*4 x 4
        x = self.head(x)
        #x: bs x n
        return x


#################
# TRAINING LOOP #
#################

writer = SummaryWriter(f'/spell/tensorboards/model_1')

train_df, test_df = get_train_test_split()
train_dataset, test_dataset = PandaDataset(train_df), PandaDataset(test_df)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = PandaModel()
model.cuda()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(1, NUM_EPOCHS + 1):
    # train
    losses = []
    model.train()
    
    for i, (X, _, y) in enumerate(iter(train_dataloader)):
        optimizer.zero_grad()

        X = X.cuda()
        y = y.cuda()

        X_pred = model(X)
        loss = criterion(X_pred, y)
        loss.backward()
        optimizer.step()

        curr_loss = loss.item()
        if i % 10 == 0:
            print(
                f'Finished training epoch {epoch}, batch {i}. Loss: {curr_loss:.3f}.'
            )

        writer.add_scalar(
            'training loss', curr_loss, epoch * len(dataloader) + i
        )
        losses.append(curr_loss)

    print(
        f'Finished training epoch {epoch}. '
        f'avg loss: {np.mean(losses)}; median loss: {np.min(losses)}'
    )
    
    if not os.path.exists('/spell/checkpoints/'):
        os.mkdir('/spell/checkpoints/')
    torch.save(model.state_dict(), f'/spell/checkpoints/model_{epoch}.pth')
    
    # validation
    losses = []
    model.eval()
    
    with torch.no_grad():
        for i, (X, _, y) in enumerate(iter(test_dataloader)):
            X = X.cuda()
            y = y.cuda()
            
            X_pred = model(X)
            loss = criterion(X_pred, y)
            curr_loss = loss.item()
            if i % 10 == 0:
                print(
                    f'Finished eval epoch {epoch}, batch {i}. Loss: {curr_loss:.3f}.'
                )
            
            writer.add_scalar(
                'validation loss', curr_loss, epoch * len(dataloader) + i
            )
            losses.append(curr_loss)

        print(
            f'Finished training epoch {epoch}. '
            f'avg loss: {np.mean(losses)}; median loss: {np.min(losses)}'
        )
