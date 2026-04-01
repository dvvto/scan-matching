import sys
sys.path.append('/tmp/sonata')

import numpy as np
import torch
import torch.nn as nn
from time import time

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

import sonata


# ScanNet 20-class metadata
SCANNET_CLASS_LABELS_20 = (
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator",
    "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
)

SCANNET_VALID_CLASS_IDS_20 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 14, 16, 24, 28, 33, 34, 36, 39,
)

SCANNET_COLOR_MAP_20 = {
    0:  (174.0, 199.0, 232.0),  # wall
    1:  (152.0, 223.0, 138.0),  # floor
    2:  (31.0,  119.0, 180.0),  # cabinet
    3:  (255.0, 187.0, 120.0),  # bed
    4:  (188.0, 189.0,  34.0),  # chair
    5:  (140.0,  86.0,  75.0),  # sofa
    6:  (255.0, 152.0, 150.0),  # table
    7:  (214.0,  39.0,  40.0),  # door
    8:  (197.0, 176.0, 213.0),  # window
    9:  (148.0, 103.0, 189.0),  # bookshelf
    10: (196.0, 156.0, 148.0),  # picture
    11: (23.0,  190.0, 207.0),  # counter
    12: (247.0, 182.0, 210.0),  # desk
    13: (219.0, 219.0, 141.0),  # curtain
    14: (255.0, 127.0,  14.0),  # refrigerator
    15: (158.0, 218.0, 229.0),  # shower curtain
    16: (44.0,  160.0,  44.0),  # toilet
    17: (112.0, 128.0, 144.0),  # sink
    18: (227.0, 119.0, 194.0),  # bathtub
    19: (82.0,   84.0, 163.0),  # otherfurniture
}


class SegHead(nn.Module):
    """Linear segmentation head on top of Sonata backbone."""
    def __init__(self, backbone_out_channels: int, num_classes: int):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(x)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    coords = points[:, :3]
    voxel_idx = np.floor(coords / voxel_size).astype(np.int64)
    min_vals = voxel_idx.min(axis=0)
    voxel_idx -= min_vals
    ranges = voxel_idx.max(axis=0) + 1
    keys = (voxel_idx[:, 0]
            + voxel_idx[:, 1] * ranges[0]
            + voxel_idx[:, 2] * ranges[0] * ranges[1])

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    num_voxels = len(unique_keys)
    counts = np.bincount(inverse, minlength=num_voxels).astype(np.float64)

    result = np.zeros((num_voxels, points.shape[1]), dtype=np.float64)
    for dim in range(points.shape[1]):
        result[:, dim] = np.bincount(inverse,
                                     weights=points[:, dim].astype(np.float64),
                                     minlength=num_voxels)
    result /= counts[:, np.newaxis]
    return result.astype(np.float32)


class SonataInference:
    def __init__(self,
                 device: str = 'cuda:0',
                 voxel_size: float = 0.02):
        self.device = device
        self.voxel_size = voxel_size

        print('[SonataInference] Loading backbone ...')
        t0 = time()
        if FLASH_ATTN_AVAILABLE:
            self.model = sonata.model.load(
                'sonata', repo_id='facebook/sonata'
            ).to(device)
        else:
            custom_config = dict(
                enc_patch_size=[1024] * 5,
                enable_flash=False,
            )
            self.model = sonata.model.load(
                'sonata',
                repo_id='facebook/sonata',
                custom_config=custom_config,
            ).to(device)
        print(f'[SonataInference] Backbone loaded in {time() - t0:.1f}s')

        print('[SonataInference] Loading segmentation head ...')
        ckpt = sonata.model.load(
            'sonata_linear_prob_head_sc',
            repo_id='facebook/sonata',
            ckpt_only=True,
        )
        self.seg_head = SegHead(**ckpt['config']).to(device)
        self.seg_head.load_state_dict(ckpt['state_dict'])
        print('[SonataInference] Segmentation head loaded. '
              f"num_classes={ckpt['config']['num_classes']}")

        self.model.eval()
        self.seg_head.eval()

        # Default transform pipeline (same as Sonata demo)
        self.transform = sonata.transform.default()

        self.class_labels = SCANNET_CLASS_LABELS_20
        self.color_map = SCANNET_COLOR_MAP_20

    def _preprocess(self, points_np: np.ndarray):
        t0 = time()
        pts = voxel_downsample(points_np, self.voxel_size)
        print(f'  downsample: {points_np.shape[0]} -> {pts.shape[0]} pts  '
              f'({time()-t0:.2f}s)')

        coords  = pts[:, :3].astype(np.float32)
        colors  = pts[:, 3:6].astype(np.float32) / 255.0  # [0,1]

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals).astype(np.float32)

        point_dict = {
            'coord':  coords,
            'color':  colors,
            'normal': normals,
        }
        point_dict = self.transform(point_dict)
        return pts, point_dict

    def _unpool_features(self, point):
        while 'pooling_parent' in point.keys():
            assert 'pooling_inverse' in point.keys()
            parent = point.pop('pooling_parent')
            inverse = point.pop('pooling_inverse')
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        feat = point.feat
        return feat, point

    def predict(self, points_np: np.ndarray):
        t_total = time()

        t1 = time()
        pts_raw, point_dict = self._preprocess(points_np)
        print(f'[SonataInference] Preprocessing: {time()-t1:.2f}s')

        t2 = time()
        for key, val in point_dict.items():
            if isinstance(val, torch.Tensor):
                point_dict[key] = val.to(self.device, non_blocking=True)

        with torch.inference_mode():
            point = self.model(point_dict)
            feat, point_top = self._unpool_features(point)  # (K, C) K = grid pts
            logits = self.seg_head(feat)                     # (K, 20)
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)

        predictions   = preds.cpu().numpy()       # (K,)
        probabilities = probs.cpu().numpy()       # (K, 20)
      
        if hasattr(point_top, 'inverse'):
            inv = point_top.inverse.cpu().numpy()
            predictions   = predictions[inv]
            probabilities = probabilities[inv]

        print(f'[SonataInference] Model forward: {time()-t2:.2f}s')
        print(f'[SonataInference] Total: {time()-t_total:.2f}s | '
              f'output shape: {predictions.shape}')

        return pts_raw, predictions, probabilities

    def get_colored_cloud(self,
                          points_downsampled: np.ndarray,
                          predictions: np.ndarray) -> np.ndarray:
        colors = np.array([self.color_map[int(p)] for p in predictions],
                          dtype=np.float32)
        result = points_downsampled.copy()
        result[:, 3:6] = colors
        return result


if __name__ == '__main__':
    import os

    bag_pcd_path = '/data/test_mssplace_by_odometry_8'
    npy_files = []
    if os.path.isdir(bag_pcd_path):
        npy_files = sorted([
            os.path.join(bag_pcd_path, f)
            for f in os.listdir(bag_pcd_path)
            if f.endswith('.npy')
        ])

    if npy_files:
        print(f'Loading {npy_files[0]}')
        pts = np.load(npy_files[0])
        print(f'Point cloud shape: {pts.shape}')
    else:
        print('No .npy files found, generating synthetic data ...')
        rng = np.random.default_rng(42)
        N = 50000
        pts = np.zeros((N, 6), dtype=np.float32)
        pts[:, :3] = rng.uniform(-5, 5, (N, 3)).astype(np.float32)
        pts[:, 3:6] = rng.integers(0, 255, (N, 3)).astype(np.float32)

    inference = SonataInference(device='cuda:0', voxel_size=0.02)
    pts_down, preds, probs = inference.predict(pts)

    print(f'\nResults:')
    print(f'  Downsampled cloud: {pts_down.shape}')
    print(f'  Predictions shape: {preds.shape}')
    print(f'  Unique classes:    {np.unique(preds)}')
    class_counts = np.bincount(preds, minlength=20)
    for i, (label, count) in enumerate(zip(SCANNET_CLASS_LABELS_20, class_counts)):
        if count > 0:
            print(f'    [{i:2d}] {label:<20s}: {count:6d} pts '
                  f'({count/len(preds)*100:.1f}%)')
