# TBD           脑肿瘤推断
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityd, EnsureTyped
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from model import UNet  # 你的3D U-Net模型构建函数
from misc import brats_post_processing  # 你之前的后处理函数
from configs import *
from blocks import PlainBlock, ResidualBlock

args = parse_seg_args()
block_dict = {
    'plain': PlainBlock,
    'res': ResidualBlock
}

kwargs = {
    "input_channels": args.input_channels,
    "output_classes": args.num_classes,   # = 3
    "channels_list": args.channels_list,
    "deep_supervision": args.deep_supervision,
    "ds_layer": args.ds_layer,
    "kernel_size": args.kernel_size,
    "dropout_prob": args.dropout_prob,
    "norm_key": args.norm,
    "block": block_dict[args.block],
}

class BrainTumor3DUNet:
    def __init__(self, model_path, device=None, patch_size=128, sw_batch_size=1, overlap=0.5):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Load model
        self.model = UNet()  # 初始化你的3D U-Net
        self._load_model_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 推理参数
        self.patch_size = patch_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap

        # 数据预处理
        self.transform = Compose([
            LoadImaged(keys=['flair', 't1', 't1ce', 't2']),
            EnsureChannelFirstd(keys=['flair', 't1', 't1ce', 't2']),
            Orientationd(keys=['flair', 't1', 't1ce', 't2'], axcodes='RAS'),
            ScaleIntensityd(keys=['flair', 't1', 't1ce', 't2']),
            EnsureTyped(keys=['flair', 't1', 't1ce', 't2']),
        ])

    def _load_model_weights(self, model_path):
        """加载预训练模型权重."""
        try:
            state = torch.load(model_path, map_location=self.device)
            if 'model' in state:
                self.model.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state)
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise e

    def predict(self, case_dict):
        """
        Args:
            case_dict (dict): {'flair': path, 't1': path, 't1ce': path, 't2': path}
        Returns:
            seg_map (np.ndarray): 预测的分割 mask, shape: (D,H,W)
        """
        try:
            # 数据预处理
            data = self.transform(case_dict)
            image = torch.cat([data[k].unsqueeze(0) for k in ['flair', 't1', 't1ce', 't2']], dim=0)  # (C,D,H,W)
            image = image.unsqueeze(0).to(self.device)  # (1,C,D,H,W)

            with torch.no_grad():
                seg_map = sliding_window_inference(
                    inputs=image,
                    predictor=self.model,
                    roi_size=(self.patch_size,) * 3,
                    sw_batch_size=self.sw_batch_size,
                    overlap=self.overlap,
                    mode='gaussian'
                )

            # 处理输出
            if seg_map.shape[1] > 1:
                seg_map = torch.argmax(seg_map, dim=1)  # 多类别
            else:
                seg_map = (seg_map > 0.5).long()  # 二分类

            seg_map = seg_map.squeeze(0).cpu().numpy()
            seg_map = brats_post_processing(seg_map)  # 后处理
            return seg_map

        except Exception as e:
            self.logger.error(f"Error during 3D U-Net prediction: {str(e)}")
            return None


# ======================
# Example usage
# ======================
# if __name__ == "__main__":
#     model_path = './checkpoints/best_model.pth'
#     case_dict = {
#         'flair': './data/Case_00001/Case_00001_flair.nii.gz',
#         't1': './data/Case_00001/Case_00001_t1.nii.gz',
#         't1ce': './data/Case_00001/Case_00001_t1ce.nii.gz',
#         't2': './data/Case_00001/Case_00001_t2.nii.gz',
#     }
#     classifier = BrainTumor3DUNet(model_path)
#     seg_map = classifier.predict(case_dict)
#     print("Segmentation map shape:", seg_map.shape)
