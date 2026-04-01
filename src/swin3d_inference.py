import sys
sys.path.insert(0, '/tmp/Swin3D')
sys.path.insert(0, '/tmp/Swin3D_Task/SemanticSeg')

import numpy as np
import torch
import torch.nn as nn
from time import time

from Swin3D.models import Swin3DUNet
from MinkowskiEngine import SparseTensor


# ScanNet 20-class metadata
SCANNET_CLASS_LABELS_20 = (
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator",
    "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
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

# Swin3D-S config matching swin3D_RGBN_S.yaml from Swin3D_Task
SWIN3D_S_CONFIG = dict(
    depths=[2, 4, 9, 4, 4],
    channels=[48, 96, 192, 384, 384],
    num_heads=[6, 6, 12, 24, 24],
    window_sizes=[5, 7, 7, 7, 7],
    quant_size=4,
    up_k=3,
    drop_path_rate=0.3,
    num_classes=20,
    num_layers=5,
    stem_transformer=True,
    upsample='linear_attn',
    first_down_stride=3,
    knn_down=True,
    in_channels=9,       # RGB(3) + Normal(3) + XYZ_voxel(3)
    cRSE='XYZ_RGB_NORM',
    fp16_mode=1,
)


def estimate_normals(coords: np.ndarray, colors: np.ndarray,
                     radius: float = 0.1, max_nn: int = 30) -> np.ndarray:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 3.]))
    return np.asarray(pcd.normals).astype(np.float32)


def data_prepare(points_np: np.ndarray, voxel_size: float = 0.02):
    """
    Preprocessing pipeline from
    Swin3D_Task/SemanticSeg/util/data_util.py :: data_prepare_scannet_point()
    """
    coords_metric = points_np[:, :3].astype(np.float32)
    colors_255    = points_np[:, 3:6].astype(np.float32)

    # Estimate normals before voxelization
    colors_01 = colors_255 / 255.0
    normals   = estimate_normals(coords_metric, colors_01)

    # feat = [R, G, B, Nx, Ny, Nz]  (following ScanNet training format)
    feat_full = np.concatenate([colors_01, normals], axis=1).astype(np.float32)

    # Shift to origin, convert to voxel units
    coord_min  = coords_metric.min(axis=0)
    coord      = (coords_metric - coord_min) / voxel_size   # float voxel units
    int_coord  = coord.astype(np.int32)

    # Voxelise: unique voxels only (keep first occurrence like Swin3D training)
    ranges  = int_coord.max(axis=0).astype(np.int64) + 1
    keys    = (int_coord[:, 0].astype(np.int64)
               + int_coord[:, 1].astype(np.int64) * ranges[0]
               + int_coord[:, 2].astype(np.int64) * ranges[0] * ranges[1])
    _, unique_map = np.unique(keys, return_index=True)

    coord     = coord[unique_map]       # (M, 3) float voxel
    feat_full = feat_full[unique_map]   # (M, 6)

    coord_t = torch.FloatTensor(coord)
    feat_t  = torch.FloatTensor(feat_full)
    return coord_t, feat_t, unique_map


class Swin3DInference:
    DEFAULT_CKPT = '/data/weights/swin3d_scannet/swin3d_s_scannet.pth'

    def __init__(self,
                 checkpoint_path: str = DEFAULT_CKPT,
                 device: str = 'cuda:0',
                 voxel_size: float = 0.02,
                 use_amp: bool = True):
        self.device     = device
        self.voxel_size = voxel_size
        self.use_amp    = use_amp
        self.class_labels = SCANNET_CLASS_LABELS_20
        self.color_map    = SCANNET_COLOR_MAP_20

        print('[Swin3DInference] Building Swin3DUNet ...')
        t0 = time()
        self.model = Swin3DUNet(**SWIN3D_S_CONFIG).to(device)

        print(f'[Swin3DInference] Loading checkpoint: {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        clean_sd = {k.replace('module.', '', 1): v
                    for k, v in state_dict.items()}
        self.model.load_state_dict(clean_sd, strict=True)
        self.model.eval()

        best_iou = ckpt.get('best_iou', float('nan'))
        epoch    = ckpt.get('epoch', '?')
        print(f'[Swin3DInference] Ready.  epoch={epoch}  '
              f'best_val_mIoU={best_iou:.4f}  ({time()-t0:.1f}s)')

    def _build_sparse_tensors(self,
                              coord: torch.Tensor,
                              feat: torch.Tensor):
        M     = coord.shape[0]
        batch = torch.zeros(M, dtype=torch.int32, device=self.device)

        feat_xyz = torch.cat([feat, coord], dim=1)    # (M, 9)

        coords_int = torch.cat(
            [batch.unsqueeze(-1), coord.int()], dim=-1
        ).int()                                        # (M, 4)

        sp = SparseTensor(feat_xyz.float(), coords_int, device=self.device)

        colors  = feat[:, 0:3] / 1.001
        normals = feat[:, 3:6] / 1.001

        coords_batch = batch.float().unsqueeze(-1)     # (M, 1)
        coords_full  = torch.cat(
            [coords_batch, coord, colors, normals], dim=1
        )                                              # (M, 10)

        coords_sp = SparseTensor(
            features=coords_full,
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )
        return sp, coords_sp

    def predict(self, points_np: np.ndarray):
        t_total = time()

        t1 = time()
        coord, feat, unique_map = data_prepare(points_np, self.voxel_size)
        M = coord.shape[0]
        print(f'[Swin3DInference] Preprocess: {points_np.shape[0]} -> {M} pts  '
              f'({time()-t1:.2f}s)')

        pts_voxelized = points_np[unique_map].copy()

        coord = coord.to(self.device)
        feat  = feat.to(self.device)

        t2 = time()
        sp, coords_sp = self._build_sparse_tensors(coord, feat)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(sp, coords_sp)

        probs = torch.softmax(output.float(), dim=-1)
        preds = output.argmax(dim=-1)

        predictions   = preds.cpu().numpy()
        probabilities = probs.cpu().numpy()

        if predictions.shape[0] != M:
            print(f'[Swin3DInference] WARNING: ME output size '
                  f'{predictions.shape[0]} != voxelized size {M}. '
                  f'Truncating to minimum.')
            n = min(predictions.shape[0], M)
            predictions   = predictions[:n]
            probabilities = probabilities[:n]
            pts_voxelized = pts_voxelized[:n]

        print(f'[Swin3DInference] Forward: {time()-t2:.2f}s | '
              f'Total: {time()-t_total:.2f}s | '
              f'Output: {predictions.shape}')
        return pts_voxelized, predictions, probabilities

    def get_colored_cloud(self,
                          pts_voxelized: np.ndarray,
                          predictions: np.ndarray) -> np.ndarray:
        colors = np.array([self.color_map[int(p)] for p in predictions],
                          dtype=np.float32)
        result = pts_voxelized.copy()
        result[:, 3:6] = colors
        return result


if __name__ == '__main__':
    import os

    data_dir = '/data/test_mssplace_by_odometry_8/place_recognition_data'
    pts = None
    if os.path.isdir(data_dir):
        for entry in sorted(os.listdir(data_dir)):
            cloud_path = os.path.join(data_dir, entry, 'cloud.npz')
            if os.path.exists(cloud_path):
                pts = np.load(cloud_path)['arr_0']
                print(f'Loaded real cloud: {cloud_path}  shape={pts.shape}')
                break

    if pts is None:
        print('No real data found, generating synthetic cloud ...')
        rng = np.random.default_rng(42)
        N = 50000
        pts = np.zeros((N, 6), dtype=np.float32)
        pts[:, :3] = rng.uniform(-5, 5, (N, 3)).astype(np.float32)
        pts[:, 3:6] = rng.integers(0, 255, (N, 3)).astype(np.float32)

    inf = Swin3DInference(device='cuda:0', voxel_size=0.02)
    pts_down, preds, probs = inf.predict(pts)

    print('\nResults:')
    print(f'  Voxelized cloud  : {pts_down.shape}')
    print(f'  Predictions shape: {preds.shape}')
    print(f'  Unique classes   : {np.unique(preds)}')
    counts = np.bincount(preds, minlength=20)
    for i, (label, count) in enumerate(zip(SCANNET_CLASS_LABELS_20, counts)):
        if count > 0:
            print(f'  [{i:2d}] {label:<20s}: {count:6d} pts '
                  f'({count / len(preds) * 100:.1f}%)')
