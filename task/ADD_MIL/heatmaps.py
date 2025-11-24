import os
import sys
import h5py
import torch
from tqdm import tqdm
from trident import OpenSlideWSI, visualize_heatmap


current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)


from config import Config
from modules.ADD_MIL.add_mil import AddMIL


class HeatmapManager(Config):
    def __init__(self):
        super().__init__()
        self.model = AddMIL()
        self.model.load_state_dict(torch.load(self.attn_model_checkpoint))
        self.model.to(self.device)

    def run_once(self, slide_id, heatmap_output_dir):
        self.model.eval()

        svs_path = os.path.join(self.heatmaps_svs, slide_id + ".svs")
        h5_path = os.path.join(self.heatmaps_features, slide_id + ".h5")

        with h5py.File(h5_path, 'r') as f:
            # 图像块的坐标
            coords = f['coords'][:]
            # 图像块特征
            patch_features = f['features'][:]
            # 坐标的属性
            coords_attrs = dict(f['coords'].attrs)

        patch_features = torch.from_numpy(patch_features).to(self.device)
        # 添加 batch=1 的维度，变为 (1, L, D)
        patch_features = patch_features.unsqueeze(0)
        with torch.no_grad():
            A_ori, patch_contributions, logits = self.model(patch_features)
            print("A_ori.shape:", A_ori.shape)
            print("patch_contributions.shape:", patch_contributions.shape)
            print("logits.shape:", logits.shape)

        slide = OpenSlideWSI(slide_path=svs_path, lazy_init=False)
        normal_heatmap_filename = f"{slide_id}_normal_heatmap.png"
        tumor_heatmap_filename = f"{slide_id}_tumor_heatmap.png"
        # .squeeze() 去除 NumPy 数组中所有维度大小为 1 的维度
        normal_scores = patch_contributions[:, :, 0].cpu().numpy().squeeze()
        tumor_scores = patch_contributions[:, :, 1].cpu().numpy().squeeze()

        visualize_heatmap(
            wsi=slide,
            scores=normal_scores,
            coords=coords,
            patch_size_level0=coords_attrs["patch_size_level0"],
            vis_level=1,
            normalize=self.normalize,
            output_dir=heatmap_output_dir,
            filename=normal_heatmap_filename
        )
        visualize_heatmap(
            wsi=slide,
            scores=tumor_scores,
            coords=coords,
            patch_size_level0=coords_attrs["patch_size_level0"],
            vis_level=1,
            normalize=self.normalize,
            output_dir=heatmap_output_dir,
            filename=tumor_heatmap_filename
        )



    def run(self):
        # 获取所有 svs 文件
        svs_files = []
        for file in os.listdir(self.heatmaps_svs):
            if file.endswith('.svs') and not file.startswith('normal') and not file.startswith('tumor'):
                svs_files.append(file)
        print(f"共找到 {len(svs_files)} 个待生成热图的 svs 文件")

        # 创建输出目录
        if self.normalize == 'rank':
            heatmaps_output_dir = os.path.join(self.heatmaps_output, f"rank_normalize")
        elif self.normalize == 'minmax':
            heatmaps_output_dir = os.path.join(self.heatmaps_output, f"minmax_normalize")
        else:
            heatmaps_output_dir = os.path.join(self.heatmaps_output, f"no_normalize")
        os.makedirs(heatmaps_output_dir, exist_ok=True)

        # 记录失败的文件
        failed_slides = []

        # 循环处理
        # desc="生成热力图": 进度条左侧的静态标题
        pbar = tqdm(svs_files, desc="生成热力图")
        for file_name in pbar:
            slide_id = os.path.splitext(file_name)[0]
            # 动态更新进度条右侧的描述，显示当前正在处理哪个 slide
            pbar.set_description(f"正在处理: {slide_id}")
            
            try:
                # 执行处理逻辑
                self.run_once(slide_id, heatmaps_output_dir)
                
                # 处理成功后清理 GPU 缓存，释放内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError as e:
                # 捕获 CUDA OOM 错误
                print(f"\n[错误] {slide_id} 处理失败: CUDA 内存不足")
                print(f"错误详情: {str(e)}")
                failed_slides.append((slide_id, "CUDA OOM"))
                
                # 清理 GPU 缓存，尝试释放内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("已清理 GPU 缓存，继续处理下一个图像...")
                    
            except Exception as e:
                # 捕获其他异常
                print(f"\n[错误] {slide_id} 处理失败: {type(e).__name__}")
                print(f"错误详情: {str(e)}")
                failed_slides.append((slide_id, f"{type(e).__name__}: {str(e)}"))
                
                # 清理 GPU 缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("已清理 GPU 缓存，继续处理下一个图像...")

        print("\n" + "="*80)
        print("处理完成汇总:")
        print(f"总共处理: {len(svs_files)} 个文件")
        print(f"成功: {len(svs_files) - len(failed_slides)} 个")
        print(f"失败: {len(failed_slides)} 个")
        
        if failed_slides:
            print("\n失败的文件列表:")
            for slide_id, reason in failed_slides:
                print(f"  - {slide_id}: {reason}")
            
            # 保存失败列表到文件
            failed_log_path = os.path.join(heatmaps_output_dir, "failed_slides.txt")
            with open(failed_log_path, 'w') as f:
                f.write("Failed Slides Log\n")
                f.write("="*80 + "\n\n")
                for slide_id, reason in failed_slides:
                    f.write(f"{slide_id}\t{reason}\n")
            print(f"\n失败列表已保存到: {failed_log_path}")
        
        print("="*80)


if __name__ == "__main__":
    heatmap_manager = HeatmapManager()
    heatmap_manager.run()
