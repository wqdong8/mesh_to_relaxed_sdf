import os
import open3d as o3d
import numpy as np
import argparse
import gc
import trimesh
import fpsample
from pysdf import SDF

def save_vertices_as_ply_open3d(vertices, filepath):
    """Save point cloud vertices as PLY file using Open3D."""
    points = vertices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector((points+1)/2)
    o3d.io.write_point_cloud(filepath, point_cloud, write_ascii=True)

def main(mesh_path, mesh_id, point_number, save_dir, debug=False) -> None:
    os.makedirs(save_dir, exist_ok=True)
    npz_output_path = os.path.join(save_dir, "sample" + ".npz")
    try:
        # Load mesh without processing to preserve original structure
        mesh = trimesh.load(mesh_path, process=False)
        
        # Sample points on the surface
        coarse_surface_points, faces = mesh.sample(point_number, return_index=True)
        normals = mesh.face_normals[faces]
        coarse_surface = np.concatenate([coarse_surface_points, normals], axis=1)
        
        # Generate near-surface points with different noise levels
        coarse_near_surface_points = []
        
        # First set with moderate noise (scale=0.01)
        resampled_points1, faces1 = mesh.sample(len(coarse_surface_points), return_index=True)
        coarse_near_surface_points.append(
            resampled_points1 + np.random.normal(scale=0.01, size=(len(resampled_points1), 3))
        )
        
        # Second set with smaller noise (scale=0.005)
        resampled_points2, faces2 = mesh.sample(len(coarse_surface_points), return_index=True)
        coarse_near_surface_points.append(
            resampled_points2 + np.random.normal(scale=0.005, size=(len(resampled_points2), 3))
        )
        
        # Concatenate all near-surface points
        coarse_near_surface_points = np.concatenate(coarse_near_surface_points)
        
        # Sample points in space (slightly beyond normalized bounds)
        space_points = np.random.uniform(-1.05, 1.05, (point_number, 3))
        rand_points = np.concatenate([coarse_near_surface_points, space_points], axis=0)
        
        # Calculate SDF values for all random points
        f = SDF(mesh.vertices, mesh.faces)
        coarse_sdf = f(rand_points).reshape(-1, 1)
        rand_points = np.concatenate([rand_points, coarse_sdf], axis=1)
        rand_indices = np.random.permutation(rand_points.shape[0])
        rand_points = rand_points[rand_indices]
        
        # Perform FPS (Farthest Point Sampling) on surface points
        fps_coarse_surface_list = []
        for _ in range(1):
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(coarse_surface_points, point_number, h=5)
            fps_coarse_surface = coarse_surface[kdline_fps_samples_idx].reshape(-1, 1, 6)
            fps_coarse_surface_list.append(fps_coarse_surface)
        fps_coarse_surface = np.concatenate(fps_coarse_surface_list, axis=1)
        
        # Clean up invalid values
        fps_coarse_surface[np.isinf(fps_coarse_surface)] = 1
        fps_coarse_surface[np.isnan(fps_coarse_surface)] = 1
        
        # Split the data into multiple NPZ files if requested
        num_splits = args.num_split

        if num_splits > 1:
            points_per_file = len(rand_points) // num_splits
            surface_per_file = fps_coarse_surface.shape[0] // num_splits
            
            for i in range(num_splits):
                # Calculate start and end indices for each split
                rand_start = i * points_per_file
                rand_end = (i + 1) * points_per_file if i < num_splits - 1 else len(rand_points)
                
                surface_start = i * surface_per_file
                surface_end = (i + 1) * surface_per_file if i < num_splits - 1 else fps_coarse_surface.shape[0]
                
                # Create split file path
                split_npz_path = npz_output_path.replace('.npz', f'_{i}.npz')
                
                # Save the split data
                np.savez(
                    split_npz_path,
                    fps_coarse_surface=fps_coarse_surface[surface_start:surface_end].astype(np.float32),
                    rand_points=rand_points[rand_start:rand_end].astype(np.float32),
                )
                
                print(f"Saved split {i+1}/{num_splits} to {split_npz_path}")
        else:
            # Save all data to a single file
            np.savez(
                npz_output_path.replace('.npz', f'_0.npz'),
                fps_coarse_surface=fps_coarse_surface.astype(np.float32),
                rand_points=rand_points.astype(np.float32),
            )
        
        # Save debug visualization if enabled
        if debug:
            os.makedirs(os.path.join(save_dir, 'debug'), exist_ok=True)
            ply_output_path = os.path.join(save_dir, 'debug', mesh_id + ".ply")
            save_vertices_as_ply_open3d(coarse_surface_points, ply_output_path)

        return True
    except Exception as e:
        print(f"ERROR: in processing path: {mesh_path}. Error: {e}")
        gc.collect()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh_path",
        type=str,
        help="input watertight mesh path",
    )
    parser.add_argument(
        "--mesh_id",
        type=str,
        help="mesh id",
    )
    parser.add_argument(
        "--point_number",
        type=int,
        default=200000,
        help="number of points to sample",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="result save dir",
    )
    parser.add_argument(
        "--num_split",
        type=int,
        default=1,
        help="number of split for sampling",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="if debug is true, save ply file, otherwise only save npz file",
    )
    args, extras = parser.parse_known_args()
    main(args.mesh_path, args.mesh_id, args.point_number, args.save_dir, args.debug)
