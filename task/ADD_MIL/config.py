import torch
import os
import sys


current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)


from utils.dataload_utils import get_dataloader


class Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 3906

        # region 数据集相关参数
        features_path = "/data/med/FEATURES_DIRECTORY/GDC-MMR-50-HE-SVS/20x_256px_0px_overlap/features_uni_v1"
        filter_path = "../../datasets/GDC-MMR-50-HE_filtered.csv"
        self.train_batch_size = 1
        self.test_batch_size = 1
        self.train_shuffle = True
        self.test_shuffle = False

        validate_features_path = "/data/med/FEATURES_DIRECTORY/MMR-18-HE-SVS/20x_256px_0px_overlap/features_uni_v1"
        validate_filter_path = "../../datasets/MMR-18-HE_filtered.csv"        
        self.validate_batch_size = 1
        self.validate_shuffle = False

        self.train_loader = get_dataloader(
            features_path=features_path, 
            filter_path=filter_path,
            split="train",
            batch_size=self.train_batch_size, 
            shuffle=self.train_shuffle,
            seed=self.seed,
        )
        self.test_loader = get_dataloader(
            features_path=features_path, 
            filter_path=filter_path,
            split="test",
            batch_size=self.test_batch_size, 
            shuffle=self.test_shuffle,
            seed=self.seed,
        )
        self.validate_loader = get_dataloader(
            features_path=validate_features_path, 
            filter_path=validate_filter_path,
            split="validate",
            batch_size=self.validate_batch_size, 
            shuffle=self.validate_shuffle,
            seed=self.seed,
        )
        # endregion

        # region 训练相关参数
        self.num_epochs = 200
        self.checkpoint_dir = "../../outputs/ADD_MIL"
        self.checkpoint_prefix = "gdc_mmr_50_he_model"
        # endregion

        # region 注意力图相关参数
        self.attn_model_checkpoint = "../../outputs/ADD_MIL/gdc_mmr_50_he_model/epoch200_20251124_213445.pth"
        self.normalize = 'rank'  # 归一化方式
        self.heatmaps_output = f"../../outputs/ADD_MIL/{self.checkpoint_prefix}/heatmaps"
        self.heatmaps_svs = "/data/med/stomach/GDC-MMR-50-HE-SVS/"
        self.heatmaps_features = "/data/med/FEATURES_DIRECTORY/GDC-MMR-50-HE-SVS/20x_256px_0px_overlap/features_uni_v1/"
        # endregion