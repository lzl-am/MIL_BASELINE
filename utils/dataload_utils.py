import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import numpy as np
import pandas as pd


class H5Dataset(Dataset):
    def __init__(self, feats_path, df, split=None):
        """
        初始化数据集
        根据 split 过滤数据，支持 "train"、"test"、"validate" 或全量读取（split=None）
        """
        # TODO 后续可进行优化
        # 根据 fold_0 列划分数据
        # 全量读取模式（不过滤）
        if split is None:
            self.df = df.copy()
        # 按指定split过滤
        else:
            self.df = df[df["fold_0"] == split].copy()
            
        self.feats_path = feats_path
        self.split = split

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.df)

    def __getitem__(self, idx):
        """加载单个样本的特征和标签"""
        row = self.df.iloc[idx]
        # 直接使用所有特征
        with h5py.File(os.path.join(self.feats_path, row['slide_id'] + '.h5'), "r") as f:
            features = torch.from_numpy(f["features"][:])
            
        label = torch.tensor(row["tumor"], dtype=torch.long)
        return features, label


def get_dataloader(
        features_path, 
        filter_path, 
        split=None,
        batch_size=1, 
        shuffle=True,
        seed=8,
    ):
        df = pd.read_csv(filter_path)
        return  DataLoader(H5Dataset(features_path, df, split), batch_size=batch_size, shuffle=shuffle, worker_init_fn=lambda _: np.random.seed(seed))
