"""
This is the first major src file that I use to keep the tuning workflow as clean and iterative as possible. 
Once the .e57 point cloud is loaded, this file contains a number of functions to voxelize the point cloud and draw out major planes.
This file also contains a function for classifying gaps as doors or not, and tweaking the parameters for the algorithms via functions is very helpful.
Modularizing the workflow like this helps keep the main notebook clean for testing while still being easy to tweak and customize.
"""

import numpy as np
import open3d as o3d


def downsample_point_cloud(points_3d: np.ndarray, voxel_size: float = 0.05):
    # Initialize the Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Load NumPy points into Open3D format
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Voxel downsample to 5cm chunks
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    print(f"Original point cloud size: {len(pcd.points):,}")
    print(f"Downsampled point cloud size: {len(pcd_down.points):,}")

    return pcd_down



def extract_structural_planes(
    pcd_down,
    voxel_size: float = 0.05,
    min_plane_area_m2: float = 0.6,
    min_horizontal_area: float = 30.0,
    max_skips: int = 20,
):
    """
    Extracts structural planes (walls, floors, ceilings) from a point cloud using iterative RANSAC.

    Args:
        pcd_down (o3d.geometry.PointCloud): Downsampled Open3D point cloud.
        voxel_size (float, optional): Voxel size in meters used for area estimations. Defaults to 0.05.
        min_plane_area_m2 (float, optional): Minimum required area in square meters for a valid plane. Defaults to 0.6.
        min_horizontal_area (float, optional): Minimum area in square meters required for horizontal surfaces. Defaults to 30.0.
        max_skips (int, optional): Maximum consecutive skipped planes allowed before exiting. Defaults to 20.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: 
            - List of plane equations [A, B, C, D].
            - List of NumPy arrays of shape (N, 3) containing points for each detected plane.
    """
    active_pcd = o3d.geometry.PointCloud(pcd_down)
    area_per_point = voxel_size * voxel_size
    min_points_threshold = int(min_plane_area_m2 / area_per_point)

    all_plane_equations = []
    all_plane_clouds = []

    iteration = 0
    consecutive_skips = 0

    while True:
        if len(active_pcd.points) < min_points_threshold or consecutive_skips > max_skips:
            break

        plane_model, inliers = active_pcd.segment_plane(
            distance_threshold=voxel_size,
            ransac_n=3,
            num_iterations=5000,
        )

        A, B, C, D = plane_model
        num_points = len(inliers)
        estimated_area = num_points * area_per_point

        is_horizontal = abs(C) > 0.7

        # Reject minor horizontal surfaces (furniture / stratified slice)
        if is_horizontal and estimated_area < min_horizontal_area:
            consecutive_skips += 1
            pts = np.asarray(active_pcd.points)
            jitter = np.random.normal(0, 0.0001, size=pts.shape)
            active_pcd.points = o3d.utility.Vector3dVector(pts + jitter)
            continue

        if estimated_area >= min_plane_area_m2:
            iteration += 1
            consecutive_skips = 0

            all_pts_matrix = np.asarray(active_pcd.points)
            detected_plane_pts = all_pts_matrix[inliers]

            all_plane_equations.append(plane_model)
            all_plane_clouds.append(detected_plane_pts)

            orientation_label = "Horizontal" if is_horizontal else "Vertical Wall"
            print(
                f"Plane #{iteration} [{orientation_label}]: Area = {estimated_area:.2f} m² ({num_points:,} pts)"
            )

            active_pcd = active_pcd.select_by_index(inliers, invert=True)
        else:
            break

    print(f"Extraction complete! Saved {iteration} high-confidence structural planes.")
    return all_plane_equations, all_plane_clouds, active_pcd



def reconstruct_walls_and_doors(
    all_plane_clouds: list[np.ndarray],
    all_plane_equations: list[np.ndarray],
    gap_max_empty_space_m: float = 0.80,
    door_height_threshold_m: float = 0.30,
):
    """
    Reconstructs angled wall panels and detects doorways from extracted plane clouds.

    Args:
        all_plane_clouds (list[np.ndarray]): List of point arrays for each extracted plane.
        all_plane_equations (list[np.ndarray]): List of plane equations [A, B, C, D].
        gap_max_empty_space_m (float, optional): Maximum gap distance before splitting 
            a wall into separate segments. Defaults to 0.80.
        door_height_threshold_m (float, optional): Maximum distance from the floor baseline 
            for a gap to be classified as a doorway. Defaults to 0.30.

    Returns:
        tuple[list[dict], list[np.ndarray], float]:
            - List of wall dictionaries containing 'corners' (4x3 array) and 'wall_idx'.
            - List of 3D points representing detected doorway centers.
            - Evaluated floor baseline height (Z value).
    """
    # Establish floor baseline
    horizontal_slab_heights = []
    for eq in all_plane_equations:
        A, B, C, D = eq
        if abs(C) > 0.7:
            horizontal_slab_heights.append(-D / C)
    floor_z = min(horizontal_slab_heights) if len(horizontal_slab_heights) > 0 else 0.0

    ### Note to self: magically solve arbitrary thresholds
    GAP_MAX_EMPTY_SPACE_M = gap_max_empty_space_m  
    DOOR_HEIGHT_THRESHOLD_M = door_height_threshold_m 

    segmented_wall_rectangles = [] 
    detected_doorways = []         

    for wall_idx, wall_pts in enumerate(all_plane_clouds):
        A, B, C, D = all_plane_equations[wall_idx]
        if abs(C) > 0.7 or len(wall_pts) < 50:
            continue
            
        # Run PCA on the wall's horizontal footprint
        xy_pts = wall_pts[:, :2]
        xy_mean = np.mean(xy_pts, axis=0)
        centered_xy = xy_pts - xy_mean
        covariance_matrix = np.cov(centered_xy.T)
        
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        wall_dir = eigenvectors[:, np.argmax(eigenvalues)] 
        
        # Project to local 2D coordinates (Length, Height)
        lengths = centered_xy @ wall_dir
        heights = wall_pts[:, 2]
        
        # Global height ceiling for this specific wall to keep the top edge flat
        global_wall_max_z = heights.max()
        
        # Sort left-to-right along the wall's true face
        sort_idx = np.argsort(lengths)
        sorted_lengths = lengths[sort_idx]
        sorted_3d_pts = wall_pts[sort_idx]
        
        # Identify gaps
        gaps = np.diff(sorted_lengths)
        split_indices = np.where(gaps > GAP_MAX_EMPTY_SPACE_M)[0]
        
        # Index gaps
        start_idx = 0
        wall_segments = []
        for split in split_indices:
            wall_segments.append((start_idx, split))
            start_idx = split + 1
        wall_segments.append((start_idx, len(sorted_lengths) - 1))
        
        # Process each continuous segment
        for i in range(len(wall_segments)):
            seg_start, seg_end = wall_segments[i]
            if (seg_end - seg_start) < 10: 
                continue
                
            min_len = sorted_lengths[seg_start]
            max_len = sorted_lengths[seg_end]
            
            # Force vertical limits to align perfectly with the floor and a flat ceiling
            ## Note not self: does this make the most sense?
            min_z = floor_z
            max_z = global_wall_max_z
            
            # Un-project horizontal endpoints back into angled global 3D space
            pt_start_xy = xy_mean + min_len * wall_dir
            pt_end_xy = xy_mean + max_len * wall_dir
            
            corners_3d = np.array([
                [pt_start_xy[0], pt_start_xy[1], min_z], # 0: Bottom Left
                [pt_end_xy[0],   pt_end_xy[1],   min_z], # 1: Bottom Right
                [pt_end_xy[0],   pt_end_xy[1],   max_z], # 2: Top Right
                [pt_start_xy[0], pt_start_xy[1], max_z]  # 3: Top Left
            ])
            
            segmented_wall_rectangles.append({
                'corners': corners_3d,
                'wall_idx': wall_idx
            })
            
            # Analyze the gap right after this segment for doors
            if i < len(wall_segments) - 1:
                next_seg_start = wall_segments[i+1][0]
                
                gap_center_x = (sorted_3d_pts[seg_end][0] + sorted_3d_pts[next_seg_start][0]) / 2.0
                gap_center_y = (sorted_3d_pts[seg_end][1] + sorted_3d_pts[next_seg_start][1]) / 2.0
                
                local_z_min = min(sorted_3d_pts[seg_end][2], sorted_3d_pts[next_seg_start][2])
                if abs(local_z_min - floor_z) < DOOR_HEIGHT_THRESHOLD_M:
                    detected_doorways.append(np.array([gap_center_x, gap_center_y, floor_z + 1.0]))

    print(f"Generated {len(segmented_wall_rectangles)} true angled wall panels.")
    return segmented_wall_rectangles, detected_doorways, floor_z