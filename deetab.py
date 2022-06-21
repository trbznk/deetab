import torch
import torch.nn as nn
import string
import time
import pandas as pd
import os.path
import argparse

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob


NROWS = None
PIN_MEMORY = False
NUM_WORKERS = 4
PATH_TRAIN_DATA = "./data/tabular-playground-series-may-2022/train.csv"
PATH_TEST_DATA = "./data/tabular-playground-series-may-2022/test.csv"
PATH_TRAIN_DATA_CACHE = "./data/tabular-playground-series-may-2022/train.pickle"
PATH_TEST_DATA_CACHE = "./data/tabular-playground-series-may-2022/test.pickle"


def encode_f_27(df):
    char_enc = df["f_27"].str.split("", expand=True).drop([0, 11], axis=1)
    char_enc.columns = [f"f_{col+30}" for col in char_enc.columns]
    for col in char_enc.columns:
        char_enc.loc[:, col] = char_enc.loc[:, col].apply(lambda c: string.ascii_uppercase.index(c))
    df = pd.concat([df, char_enc], axis=1)
    df["f_27"] = df["f_27"].apply(lambda s: len(set(s)))
    return df


def list_models():
    paths = glob("./models/*.pt")
    for path in paths:
        checkpoint = torch.load(path)
        print(f"{path} {checkpoint['name']} train_loss={checkpoint['train_loss']:.2f} dev_loss={checkpoint['dev_loss']:.2f} epoch={checkpoint['epoch']}")


def standardize(df, mean, std):
    return (df-mean)/std


def read_tabular_playground_series_may_2022():
    if os.path.exists(PATH_TRAIN_DATA_CACHE) and os.path.exists(PATH_TEST_DATA_CACHE):
        print("INFO: read data from cache")
        train_df = pd.read_pickle(PATH_TRAIN_DATA_CACHE)
        test_df = pd.read_pickle(PATH_TEST_DATA_CACHE)
    else:
        print("INFO: read data")
        train_df = pd.read_csv(PATH_TRAIN_DATA, nrows=NROWS, index_col="id")
        test_df = pd.read_csv(PATH_TEST_DATA, nrows=NROWS, index_col="id")
        train_df = encode_f_27(train_df)
        test_df = encode_f_27(test_df)
        train_df.to_pickle(PATH_TRAIN_DATA_CACHE)
        test_df.to_pickle(PATH_TEST_DATA_CACHE)

    cols = [col for col in test_df.columns]
    target_col = "target"

    mean = train_df[cols].mean()
    std = train_df[cols].std()

    train_df[cols] = standardize(train_df[cols], mean, std)
    test_df[cols] = standardize(test_df[cols], mean, std)
    
    train_set = TabularData(train_df, target_col="target")
    train_set_size = int(len(train_set) * 0.8)
    dev_set_size = len(train_set)-train_set_size
    train_set, dev_set = torch.utils.data.random_split(train_set, [train_set_size, dev_set_size])

    test_set = TabularData(test_df)
    return train_set, dev_set, test_set


def save_model(path, model_name, epoch, mean_train_loss, mean_dev_loss, model, optimizer):
    torch.save({
        "name": model_name,
        "epoch": epoch,
        "train_loss": mean_train_loss,
        "dev_loss": mean_dev_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)


class TabularData(Dataset):
    def __init__(self, df, target_col=None):
        super().__init__()
        self.target_col = target_col
        cols = [col for col in df.columns if col != self.target_col]   
        self.X = df[cols].values
        if self.target_col is not None:
            self.Y = df[target_col].to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        if self.target_col is not None:
            y = torch.tensor(self.Y[i], dtype=torch.float32)
            return x, y
        
        return x, None


class DeeTab(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.m = nn.Sequential(
            nn.Linear(41, 512),
            nn.SiLU(),
            nn.Linear(512, 384),
            nn.SiLU(),
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.m(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_models()

    train_set, dev_set, test_set = read_tabular_playground_series_may_2022()

    batch_size = 4096
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = DeeTab()
    
    if args.train:
        print("INFO: train")
        model_name = str(hex(time.time_ns()))[2:]
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)

        epochs = 50
        best_loss = 1
        for epoch in range(epochs):        
            model.train()
            running_train_loss = 0
            for x, y in tqdm(train_loader, desc="TRAIN", total=len(train_loader)):
                optimizer.zero_grad(set_to_none=True)
                output_data = model(x)
                output_data = output_data.view(-1)
                loss = loss_fn(output_data, y)
                running_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            model.eval()
            running_dev_loss = 0
            for x, y in tqdm(dev_loader, desc="DEV", total=len(dev_loader)): 
                output_data = model(x)
                output_data = output_data.view(-1)
                loss = loss_fn(output_data, y)
                running_dev_loss += loss.item()
            
            mean_train_loss = (running_train_loss/len(train_loader))
            mean_dev_loss = (running_dev_loss/len(dev_loader))
            print(f"({epoch+1}) TRAIN: {mean_train_loss:.2f} DEV: {mean_dev_loss:.2f}")

            path = f"./models/last_{model_name}.pt"
            save_model(path, model_name, epoch+1, mean_train_loss, mean_dev_loss, model, optimizer)
            if mean_dev_loss < best_loss:
                path = f"./models/best_{model_name}.pt"
                save_model(path, model_name, epoch+1, mean_train_loss, mean_dev_loss, model, optimizer)
                best_loss = mean_dev_loss
    elif args.test:
        print("INFO: test")
