#!/usr/bin/env python3
"""
test_all_models.py
==================
Unified benchmark script that:
  1. Runs semantic segmentation inference with OctFormer, Sonata (PTv3) and
     Swin3D-S on every point cloud in the test dataset.
  2. Saves per-cloud results (points_downsampled, colors, predictions,
     probabilities) for each model into a model-specific sub-folder.
  3. Evaluates the scan-matching pipeline (semantic vs. plain-ORB) for each
     model and prints mIoU-style class-distribution statistics.
  4. Plots Precision-Recall curves for all models and saves the figure.

Usage (inside the Docker container):
    cd /workspace
    python test_all_models.py [--models octformer sonata swin3d]
                              [--skip_inference]
                              [--out_dir /data/results]
"""

import sys
import os
import argparse
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PR_DATA_DIR    = '/data/test_mssplace_by_odometry_8/place_recognition_data'
GRAPH_DATA_DIR = '/data/test_mssplace_by_odometry_8/graph_data'
BAG_PATH       = '/data/bags/habitat_point_clouds.bag'

OCTFORMER_CONFIG = '/home/docker_prism/octformer/configs/seg_scannet.yaml'
OCTFORMER_CKPT   = '/data/weights/octformer_scannet/best_model.pth'
SWIN3D_CKPT      = '/data/weights/swin3d_scannet/swin3d_s_scannet.pth'

# ---------------------------------------------------------------------------
# ScanNet-20 metadata
# ---------------------------------------------------------------------------
SCANNET_CLASSES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair',
    'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator',
    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture',
]
NUM_CLASSES = 20


def load_octformer():
    sys.path.insert(0, '/home/docker_prism/octformer')
    from octformer_inference import StandaloneOctformerInference

    inf = StandaloneOctformerInference(
        config_path=OCTFORMER_CONFIG,
        checkpoint_path=OCTFORMER_CKPT,
        device='cuda:0',
    )

    def predict(points_np):
        pts_d, colors_d, preds, probs = inf.predict(points_np)
        pts_d6 = pts_d[:, :6].astype(np.float32)
        # OctFormer colors are already [0,1]
        return pts_d6, colors_d.astype(np.float32), preds.astype(np.int32), probs.astype(np.float32)

    return predict


def load_sonata():
    sys.path.insert(0, '/tmp/sonata')
    from sonata_inference import SonataInference

    inf = SonataInference(device='cuda:0', voxel_size=0.02)

    def predict(points_np):
        pts_raw, preds, probs = inf.predict(points_np)
        pts_d = pts_raw.copy().astype(np.float32)
        pts_d[:, 3:6] /= 255.0          # normalise rgb → [0,1]
        colors_d = pts_d[:, 3:6].copy()
        return pts_d, colors_d, preds.astype(np.int32), probs.astype(np.float32)

    return predict


def load_swin3d():
    sys.path.insert(0, '/tmp/Swin3D')
    sys.path.insert(0, '/tmp/Swin3D_Task/SemanticSeg')
    from swin3d_inference import Swin3DInference

    inf = Swin3DInference(checkpoint_path=SWIN3D_CKPT, device='cuda:0', voxel_size=0.02)

    def predict(points_np):
        pts_v, preds, probs = inf.predict(points_np)
        pts_d = pts_v.copy().astype(np.float32)
        pts_d[:, 3:6] /= 255.0
        colors_d = pts_d[:, 3:6].copy()
        return pts_d, colors_d, preds.astype(np.int32), probs.astype(np.float32)

    return predict


MODEL_LOADERS = {
    'octformer': load_octformer,
    'sonata':    load_sonata,
    'swin3d':    load_swin3d,
}


def run_inference(model_name: str, predict_fn, out_dir: str):
    print(f'\n{"="*60}')
    print(f'  INFERENCE  —  {model_name.upper()}')
    print(f'{"="*60}')

    def _process_dir(src_dir, dst_dir, height_filter=1.5):
        os.makedirs(dst_dir, exist_ok=True)
        entries = sorted(os.listdir(src_dir))
        times = []
        for entry in tqdm(entries, desc=f'{model_name} / {os.path.basename(src_dir)}'):
            cloud_path = os.path.join(src_dir, entry, 'cloud.npz')
            if not os.path.exists(cloud_path):
                continue
            dst_entry = os.path.join(dst_dir, entry)
            os.makedirs(dst_entry, exist_ok=True)

            cloud = np.load(cloud_path)['arr_0']
            cloud = cloud[cloud[:, 2] < height_filter]
            if len(cloud) < 10:
                continue

            t0 = time.time()
            pts_d, colors_d, preds, probs = predict_fn(cloud)
            times.append(time.time() - t0)

            np.savez_compressed(os.path.join(dst_entry, 'points_downsampled.npz'), pts_d)
            np.savez_compressed(os.path.join(dst_entry, 'colors_downsampled.npz'),  colors_d)
            np.savez_compressed(os.path.join(dst_entry, 'predictions.npz'),         preds)
            np.savez_compressed(os.path.join(dst_entry, 'probabilities.npz'),       probs)

        if times:
            print(f'  [{model_name}] avg inference time: {np.mean(times):.2f}s  '
                  f'(min {np.min(times):.2f}  max {np.max(times):.2f})')

    model_root = os.path.join(out_dir, model_name)
    _process_dir(
        PR_DATA_DIR,
        os.path.join(model_root, 'place_recognition_data'),
    )
    _process_dir(
        GRAPH_DATA_DIR,
        os.path.join(model_root, 'graph_data'),
    )


def run_evaluation(model_name: str, out_dir: str):
    import torch
    from local_grid import LocalGrid
    from occupancy_grid import Feature2DGlobalRegistrationPipeline as OrbPipeline
    from semantic_grid_ransac import (
        Feature2DGlobalRegistrationPipeline as SemPipeline,
    )
    from utils import get_rel_pose, get_occupancy_grid, get_iou as get_iou_grids

    model_root = os.path.join(out_dir, model_name)
    pr_dir     = os.path.join(model_root, 'place_recognition_data')
    graph_dir  = os.path.join(model_root, 'graph_data')

    def normalize(a):
        while a < -np.pi: a += 2 * np.pi
        while a >  np.pi: a -= 2 * np.pi
        return a

    def transformation_error(gt_pose_shift, tf_matrix):
        from scipy.spatial.transform import Rotation
        rot  = Rotation.from_matrix(tf_matrix[:3, :3]).as_rotvec()
        gx, gy, ga = gt_pose_shift
        ex = np.abs(tf_matrix[0, 3] - gx)
        ey = np.abs(tf_matrix[1, 3] - gy)
        ea = np.abs(normalize(rot[2] - ga))
        return (ex, ey, ea)

    def get_iou(rel_x, rel_y, rel_theta, cur_cloud, v_cloud):
        cur_grid = get_occupancy_grid(cur_cloud)
        v_grid   = get_occupancy_grid(v_cloud)
        try:
            _, iou = get_iou_grids(rel_x, rel_y, rel_theta, cur_grid, v_grid)
            return iou
        except Exception:
            return 0.0

    pipeline_sem = SemPipeline(
        outlier_thresholds=[2.5, 1.0, 0.5, 0.25, 0.25],
        ransac_iterations=1000,
    )
    pipeline_orb = OrbPipeline(
        outlier_thresholds=[2.5, 1.0, 0.5, 0.25, 0.25],
    )

    semantic_results = []
    orb_results      = []
    ious             = []

    if not os.path.isdir(pr_dir):
        print(f'  [{model_name}] No inference results found at {pr_dir}. Skip.')
        return semantic_results, orb_results, ious

    entries = sorted(os.listdir(pr_dir))
    for entry in tqdm(entries, desc=f'Eval {model_name}'):
        test_dir = os.path.join(pr_dir, entry)
        orig_dir = os.path.join(PR_DATA_DIR, entry)

        transforms_path = os.path.join(orig_dir, 'transforms.txt')
        if not os.path.exists(transforms_path):
            continue
        try:
            transforms_ = np.loadtxt(transforms_path)
        except Exception:
            continue
        if transforms_.ndim == 1:
            transforms_ = transforms_[np.newaxis, :]
        if transforms_.size == 0:
            continue

        gt_poses_path = os.path.join(orig_dir, 'gt_poses.txt')
        if not os.path.exists(gt_poses_path):
            continue
        gt_poses = np.loadtxt(gt_poses_path)

        # Load ref cloud
        pts_path = os.path.join(test_dir, 'points_downsampled.npz')
        prb_path = os.path.join(test_dir, 'probabilities.npz')
        if not os.path.exists(pts_path) or not os.path.exists(prb_path):
            continue
        ref_pts  = np.load(pts_path)['arr_0'][:, :3]
        ref_prbs = np.load(prb_path)['arr_0']
        ref_pts  = ref_pts[ref_pts == ref_pts].reshape(-1, 3)

        ref_grid = LocalGrid(
            semantic_probability_threshold=0.3,
            floor_height=-0.9,
            ceil_height=1.5,
        )
        ref_grid.update_from_cloud_and_transform(ref_pts, ref_prbs)
        ref_grid_tensor   = torch.Tensor(ref_grid.layers['occupancy'])
        ref_grid_semantic = ref_grid.layers['semantic']

        for i in range(transforms_.shape[0]):
            idx = int(transforms_[i, 0])
            cand_graph_dir = os.path.join(graph_dir, str(idx))
            cand_orig_dir  = os.path.join(GRAPH_DATA_DIR, str(idx))

            cpts_path = os.path.join(cand_graph_dir, 'points_downsampled.npz')
            cprb_path = os.path.join(cand_graph_dir, 'probabilities.npz')
            pose_path = os.path.join(cand_orig_dir,  'pose_stamped.txt')
            if not all(os.path.exists(p) for p in [cpts_path, cprb_path, pose_path]):
                continue

            cand_pts  = np.load(cpts_path)['arr_0'][:, :3]
            cand_prbs = np.load(cprb_path)['arr_0']
            cand_pose = np.loadtxt(pose_path)[1:]
            cand_pts  = cand_pts[cand_pts == cand_pts].reshape(-1, 3)

            cand_grid = LocalGrid(
                semantic_probability_threshold=0.3,
                floor_height=-0.9,
                ceil_height=1.5,
            )
            cand_grid.update_from_cloud_and_transform(cand_pts, cand_prbs)
            cand_grid_tensor   = torch.Tensor(cand_grid.layers['occupancy'])
            cand_grid_semantic = cand_grid.layers['semantic']

            pose_shift = get_rel_pose(*gt_poses[0], *cand_pose)

            # Semantic pipeline 
            tf_sem, score_sem = pipeline_sem.infer(
                ref_grid_tensor, ref_grid_semantic,
                cand_grid_tensor, cand_grid_semantic,
                verbose=False,
            )
            if tf_sem is not None:
                tf_mat = ref_grid.get_tf_matrix_xy(*tf_sem)
                err_sem = transformation_error(pose_shift, np.linalg.inv(tf_mat))
            else:
                err_sem = (np.inf, np.inf, np.inf)
                score_sem = 0.0

            # Plain ORB pipeline
            tf_orb, score_orb = pipeline_orb.infer(
                ref_grid_tensor, cand_grid_tensor,
                verbose=False,
            )
            if tf_orb is not None:
                tf_mat = ref_grid.get_tf_matrix_xy(*tf_orb)
                err_orb = transformation_error(pose_shift, np.linalg.inv(tf_mat))
            else:
                err_orb = (np.inf, np.inf, np.inf)
                score_orb = 0.0

            iou = get_iou(*pose_shift, ref_pts, cand_pts)

            semantic_results.append((score_sem, err_sem))
            orb_results.append((score_orb, err_orb))
            ious.append(iou)

    print(f'  [{model_name}] evaluated {len(ious)} scan pairs')
    return semantic_results, orb_results, ious


def compute_pr_curve(results, ious, error_thresh=0.5, iou_thresh=0.25,
                     n_points=50):
    scores  = np.array([r[0] for r in results], dtype=float)
    errors  = np.array([max(r[1]) for r in results], dtype=float)
    ious_np = np.array(ious, dtype=float)

    n_positives = (ious_np >= iou_thresh).sum()
    thresholds  = np.linspace(0.0, 1.0, n_points)
    precisions  = []
    recalls     = []

    for thr in thresholds:
        retrieved = scores >= thr
        tp = ((retrieved) & (errors < error_thresh)).sum()
        fp = ((retrieved) & (errors >= error_thresh)).sum()
        fn = ((~retrieved) & (ious_np >= iou_thresh)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec  = tp / n_positives if n_positives > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    return thresholds, np.array(precisions), np.array(recalls)


def print_stats_at_threshold(model_name, pipeline_name,
                              results, ious,
                              threshold=0.7, error_thresh=0.5, iou_thresh=0.25):
    ious_np = np.array(ious)
    scores  = np.array([r[0] for r in results])
    errors  = np.array([max(r[1]) for r in results])

    n_correct = int(((scores >= threshold) & (errors < error_thresh)).sum())
    n_wrong   = int(((scores >= threshold) & (errors >= error_thresh)).sum())
    n_missed_025 = int(((scores < threshold) & (ious_np > 0.25)).sum())
    n_missed_05  = int(((scores < threshold) & (ious_np > 0.50)).sum())

    denom_p = n_correct + n_wrong
    prec = n_correct / denom_p if denom_p > 0 else float('nan')
    n_true_025 = (ious_np > 0.25).sum()
    n_true_05  = (ious_np > 0.50).sum()
    rec_025 = 1 - n_missed_025 / n_true_025 if n_true_025 > 0 else float('nan')
    rec_05  = 1 - n_missed_05  / n_true_05  if n_true_05  > 0 else float('nan')

    print(f'  [{model_name} / {pipeline_name}]  thr={threshold}')
    print(f'    Correct matches : {n_correct}')
    print(f'    Wrong   matches : {n_wrong}')
    print(f'    Precision       : {prec:.3f}')
    print(f'    Recall@IoU>0.25 : {rec_025:.3f}  ({n_missed_025} missed / {n_true_025} total)')
    print(f'    Recall@IoU>0.50 : {rec_05:.3f}  ({n_missed_05}  missed / {n_true_05}  total)')


def print_class_distribution(model_name: str, out_dir: str):
    model_root = os.path.join(out_dir, model_name)
    pr_dir     = os.path.join(model_root, 'place_recognition_data')
    if not os.path.isdir(pr_dir):
        return

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total  = 0
    for entry in os.listdir(pr_dir):
        p = os.path.join(pr_dir, entry, 'predictions.npz')
        if not os.path.exists(p):
            continue
        preds = np.load(p)['arr_0']
        for c in range(NUM_CLASSES):
            counts[c] += (preds == c).sum()
        total += len(preds)

    if total == 0:
        return
    print(f'\n  [{model_name}] Class distribution over place-recognition clouds:')
    for c, label in enumerate(SCANNET_CLASSES):
        pct = counts[c] / total * 100
        bar = '#' * int(pct / 2)
        print(f'    [{c:2d}] {label:<20s} {pct:5.1f}%  {bar}')


def plot_pr_curves(all_pr: dict, save_path: str, iou_thresh: float = 0.25):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    styles = {
        'octformer / semantic': ('C0', '-',  'OctFormer + Semantic'),
        'octformer / orb':      ('C0', '--', 'OctFormer + ORB'),
        'sonata / semantic':    ('C1', '-',  'Sonata (PTv3) + Semantic'),
        'sonata / orb':         ('C1', '--', 'Sonata (PTv3) + ORB'),
        'swin3d / semantic':    ('C2', '-',  'Swin3D-S + Semantic'),
        'swin3d / orb':         ('C2', '--', 'Swin3D-S + ORB'),
    }

    ax_pr  = axes[0]   # Precision-Recall curve
    ax_thr = axes[1]   # Precision/Recall vs threshold

    for key, (thrs, precs, recs) in all_pr.items():
        col, ls, label = styles.get(key, ('gray', '-', key))
        ax_pr.plot(recs, precs, color=col, linestyle=ls, linewidth=2, label=label)
        ax_thr.plot(thrs, precs, color=col, linestyle=ls,       linewidth=2, label=f'{label} (P)')
        ax_thr.plot(thrs, recs,  color=col, linestyle=ls, alpha=0.4, linewidth=1.5)

    ax_pr.set_xlabel('Recall',    fontsize=13)
    ax_pr.set_ylabel('Precision', fontsize=13)
    ax_pr.set_title(f'Precision-Recall curve  (IoU>{iou_thresh})', fontsize=14)
    ax_pr.set_xlim([0, 1]);  ax_pr.set_ylim([0, 1.05])
    ax_pr.legend(fontsize=9, loc='lower left')
    ax_pr.grid(True, alpha=0.3)

    ax_thr.set_xlabel('Score threshold', fontsize=13)
    ax_thr.set_ylabel('Precision / Recall', fontsize=13)
    ax_thr.set_title('P/R vs. threshold (solid=Precision, faded=Recall)', fontsize=12)
    ax_thr.set_xlim([0, 1]);  ax_thr.set_ylim([0, 1.05])
    ax_thr.legend(fontsize=9, loc='lower left')
    ax_thr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nPR curves saved → {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['octformer', 'sonata', 'swin3d'],
                        choices=list(MODEL_LOADERS.keys()),
                        help='Which models to benchmark')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference phase (use already-saved results)')
    parser.add_argument('--out_dir', default='/data/results',
                        help='Root directory for saved results')
    parser.add_argument('--iou_thresh', type=float, default=0.25,
                        help='IoU threshold to define a "true" scan loop')
    parser.add_argument('--error_thresh', type=float, default=0.5,
                        help='Max transformation error (m or rad) for a correct match')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_pr   = {}   # key → (thresholds, precisions, recalls)

    for model_name in args.models:
        if not args.skip_inference:
            print(f'\nLoading model: {model_name} ...')
            try:
                predict_fn = MODEL_LOADERS[model_name]()
            except Exception as e:
                print(f'  [WARN] Could not load {model_name}: {e}')
                continue
            run_inference(model_name, predict_fn, args.out_dir)
            del predict_fn
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()

        print_class_distribution(model_name, args.out_dir)

        sem_res, orb_res, ious = run_evaluation(model_name, args.out_dir)

        if len(ious) == 0:
            print(f'  [{model_name}] No results – skipping PR computation.')
            continue

        print()
        print_stats_at_threshold(model_name, 'semantic', sem_res, ious,
                                 threshold=0.7,
                                 error_thresh=args.error_thresh,
                                 iou_thresh=args.iou_thresh)
        print_stats_at_threshold(model_name, 'orb',      orb_res, ious,
                                 threshold=0.7,
                                 error_thresh=args.error_thresh,
                                 iou_thresh=args.iou_thresh)

        for pipeline_name, results in [('semantic', sem_res), ('orb', orb_res)]:
            thrs, precs, recs = compute_pr_curve(
                results, ious,
                error_thresh=args.error_thresh,
                iou_thresh=args.iou_thresh,
            )
            all_pr[f'{model_name} / {pipeline_name}'] = (thrs, precs, recs)

    if all_pr:
        fig_path = os.path.join(args.out_dir, 'pr_curves.png')
        plot_pr_curves(all_pr, fig_path, iou_thresh=args.iou_thresh)

        # Also save raw data
        np.save(os.path.join(args.out_dir, 'pr_data.npy'),
                {k: {'thrs': v[0], 'precs': v[1], 'recs': v[2]}
                 for k, v in all_pr.items()})
    else:
        print('\nNo PR data to plot.')


if __name__ == '__main__':
    main()
