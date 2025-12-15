import numpy as np
from scipy.ndimage import label


def brats_post_processing(seg_map: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    后处理 3D U-Net 输出的肿瘤分割 mask。

    Args:
        seg_map (np.ndarray): 预测 mask, shape = (D, H, W) 或 (N, D, H, W)
                              0=背景, 1=ET, 2=TC, 3=WT 或自定义编码
        min_size (int): 去掉小体积连通区域的最小 voxel 数

    Returns:
        np.ndarray: 后处理后的 mask, shape 同输入
    """
    # 如果输入是 batch
    if seg_map.ndim == 4:
        output = np.zeros_like(seg_map)
        for i in range(seg_map.shape[0]):
            output[i] = brats_post_processing(seg_map[i], min_size)
        return output

    # 创建输出 mask
    processed = np.zeros_like(seg_map, dtype=np.uint8)

    # 假设编码为 1=WT, 2=TC, 3=ET
    # 可以根据你的数据集自定义
    # 通常 BraTS 提供 multi-class mask, 这里做例子处理 WT/TC/ET

    # Step 1: 先去掉小体积区域
    for label_id in np.unique(seg_map):
        if label_id == 0:
            continue
        mask = seg_map == label_id
        labeled_mask, num_features = label(mask)
        for i in range(1, num_features + 1):
            component = labeled_mask == i
            if component.sum() >= min_size:
                processed[component] = label_id

    # Step 2: 修正逻辑关系（保证 ET ⊆ TC ⊆ WT）
    # 先 ET
    et_mask = processed == 3
    tc_mask = processed == 2
    wt_mask = processed == 1

    # ET 必须在 TC 内
    tc_mask = tc_mask | et_mask
    # TC 必须在 WT 内
    wt_mask = wt_mask | tc_mask

    processed[wt_mask] = 1
    processed[tc_mask] = 2
    processed[et_mask] = 3

    return processed
