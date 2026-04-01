import sys
sys.path.append('/home/docker_prism/octformer')
import numpy as np
import open3d as o3d
import yaml
import torch
import ocnn
from time import time
from thsolver.config import parse_args
from builder import get_segmentation_model

def voxel_downsample_ultrafast(points, voxel_size):
    """
    Ultra-fast downsampling using bincount and fully vectorized operations.
    This is the fastest implementation while maintaining correctness.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Input point cloud with shape (N, 6) where each row is (x, y, z, r, g, b)
    voxel_size : float
        Size of the voxel grid (d x d x d)
    
    Returns:
    --------
    numpy.ndarray
        Downsampled point cloud with shape (M, 6) where M <= N
    """
    coords = points[:, :3]
    colors = points[:, 3:].astype(np.float32)
    
    # Calculate voxel indices
    voxel_indices = np.floor(coords / voxel_size).astype(np.int64)
    
    # Normalize indices to avoid negative numbers
    min_vals = voxel_indices.min(axis=0)
    max_vals = voxel_indices.max(axis=0)
    
    normalized_indices = voxel_indices - min_vals
    ranges = max_vals - min_vals + 1
    
    # Create unique key (1D index) for each voxel
    # Using strides to create a unique 1D index
    stride_y = ranges[0]
    stride_z = ranges[0] * ranges[1]
    
    voxel_keys = (normalized_indices[:, 0] + 
                  normalized_indices[:, 1] * stride_y + 
                  normalized_indices[:, 2] * stride_z)
    
    # Get unique voxel keys and inverse indices
    unique_keys, inverse_indices = np.unique(voxel_keys, return_inverse=True)
    num_voxels = len(unique_keys)
    
    # Use bincount to get the number of points per voxel
    voxel_counts = np.bincount(inverse_indices)
    
    # Initialize arrays for sums
    coord_sums = np.zeros((num_voxels, 3), dtype=np.float64)
    color_sums = np.zeros((num_voxels, 3), dtype=np.float64)
    
    # Accumulate sums using bincount approach (vectorized)
    # This is much faster than using np.add.at for large arrays
    for dim in range(3):
        # For coordinates
        coord_sums[:, dim] = np.bincount(inverse_indices, weights=coords[:, dim].astype(np.float64))
        
        # For colors
        color_sums[:, dim] = np.bincount(inverse_indices, weights=colors[:, dim])
    
    # Compute averages
    centroids = coord_sums / voxel_counts[:, np.newaxis]
    avg_colors = color_sums / voxel_counts[:, np.newaxis]
    
    # Combine results
    result = np.hstack([centroids, avg_colors]).astype(np.float32)
    
    return result


class StandaloneOctformerInference:
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        """
        Standalone implementation that doesn't use the solver at all
        """
        self.device = device
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get model parameters
        model_config = self.config.get('MODEL', {})
        data_config = self.config.get('DATA', {}).get('test', {})
        loss_config = self.config.get('LOSS', {})
        
        self.depth = data_config.get('depth', 6)
        self.full_depth = data_config.get('full_depth', 2)
        self.feature = model_config.get('feature', 'norm')
        self.nempty = model_config.get('nempty', True)
        self.num_classes = loss_config.get('num_class', 20)
        self.scale_factor = 10.24
        
        # Import the actual model class
        # You might need to import the specific model class from builder
        sys.argv = [
            'segmentation.py', 
            '--config', config_path,
            'LOSS.mask', '-255',
            'SOLVER.gpu', '[0]',
            'SOLVER.run', 'evaluate',
            'SOLVER.ddp_mode', 'spawn',
            'SOLVER.port', '12345',
            'SOLVER.progress_bar', 'False',
            'SOLVER.use_amp', 'False',
            'SOLVER.log_per_iter', '0',
            'SOLVER.empty_cache', '0',
            'SOLVER.zero_grad_to_none', 'True',
            'SOLVER.clip_grad', '0.0',
            'SOLVER.eval_epoch', '1',
            'DATA.test.batch_size', '1'
        ]
        FLAGS = parse_args()
        self.model = get_segmentation_model(FLAGS.MODEL)
        self.model.eval()
        self.model.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def preprocess_point_cloud(self, points_np):
        """Preprocess point cloud"""
        points_downsampled = voxel_downsample_ultrafast(points_np, voxel_size=0.05)
        coords = points_downsampled[:, :3].astype(np.float32)
        colors = points_downsampled[:, 3:6].astype(np.float32) / 255.0
        # Center and normalize
        coords_center = (coords.max(axis=0) + coords.min(axis=0)) / 2
        coords = (coords - coords_center) / self.scale_factor
        return points_downsampled, coords, colors
    
    def get_input_feature(self, octree):
        """Get input features from octree"""
        octree_feature = ocnn.modules.InputFeature(self.feature, self.nempty)
        return octree_feature(octree)

    def build_input_features(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        normals = np.asarray(pcd.normals)
        points_tensor = torch.from_numpy(points).float().to(self.device)
        colors_tensor = torch.from_numpy(colors).float().to(self.device)
        normals_tensor = torch.from_numpy(normals).float().to(self.device)
        points_ocnn = ocnn.octree.Points(points=points_tensor, 
                                         normals=normals_tensor,
                                         features=colors_tensor)
        octree = ocnn.octree.Octree(self.depth, self.full_depth, device=self.device)
        octree.build_octree(points_ocnn[:, :3])
        octree.construct_all_neigh()
        octree = octree.to(self.device)
        # Get features and run model
        data = self.get_input_feature(octree)
        # Create query points
        points_obj = ocnn.octree.merge_points([points_ocnn])
        query_pts = torch.cat([points_obj.points, points_obj.batch_id], dim=1)
        return data, octree, query_pts
    
    def predict(self, points_np):
        t1 = time()
        points_downsampled, processed_points, processed_colors = self.preprocess_point_cloud(points_np)
        t2 = time()
        print('Preprocessing time:', t2 - t1)
        data, octree, query_pts = self.build_input_features(processed_points, processed_colors)
        t3 = time()
        print('Octree building time:', t3 - t2)
        with torch.no_grad():
            logits = self.model(data, octree, octree.depth, query_pts)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
        t4 = time()
        print('Inference time:', t4 - t3)
        return points_downsampled, processed_colors, predictions.cpu().numpy(), probabilities.cpu().numpy()