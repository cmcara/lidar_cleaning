import numpy as np
from scipy.spatial import cKDTree


def densify_trajectory(trajectory_points: np.ndarray, target_spacing_meters: float = 0.05) -> np.ndarray:
    """
    Takes sparse survey positions and interpolates a continuous linear 
    path trail between them at a fixed spatial cutoff.
    """
    densified_trajectory = []
    
    for i in range(len(trajectory_points) - 1):
        p_start = trajectory_points[i]
        p_end = trajectory_points[i + 1]
        
        segment_vector = p_end - p_start
        segment_distance = np.linalg.norm(segment_vector)
        
        # Determine steps needed to maintain target spacing
        num_steps = int(np.ceil(segment_distance / target_spacing_meters))
        num_steps = max(num_steps, 2)
        
        t_values = np.linspace(0, 1, num_steps)
        segment_points = p_start + t_values[:, np.newaxis] * segment_vector
        
        # Exclude the last point of the segment to prevent overlapping joint duplicates
        densified_trajectory.append(segment_points[:-1])
        
    # Append the absolute terminal waypoint of the path
    densified_trajectory.append(trajectory_points[-1:])
    return np.vstack(densified_trajectory)


def find_nearest_trajectory_waypoints(down_points: np.ndarray, trajectory_points: np.ndarray) -> np.ndarray:
    """
    Maps each point to its closest position along the continuous trajectory trail
    using a spatial KD-Tree.
    """
    trajectory_tree = cKDTree(trajectory_points)
    _, nearest_indices = trajectory_tree.query(down_points, k=1, workers=-1)
    return trajectory_points[nearest_indices]


def is_point_in_rectangle_3d(intersection_pt: np.ndarray, corners: np.ndarray) -> bool:
    """
    Verifies if a 3D intersection point lies inside the bounded 4-corner wall panel.
    """
    p0, p1, p3 = corners[0], corners[1], corners[3]
    v1 = p1 - p0
    v2 = p3 - p0
    
    v_pt = intersection_pt - p0
    
    dot1 = np.dot(v_pt, v1)
    dot2 = np.dot(v_pt, v2)
    
    in_v1 = (0 <= dot1) and (dot1 <= np.dot(v1, v1))
    in_v2 = (0 <= dot2) and (dot2 <= np.dot(v2, v2))
    
    return in_v1 and in_v2


def run_ray_casting_classification(
    down_points: np.ndarray, 
    trajectory_points: np.ndarray, 
    wall_panels: list[dict], 
    all_plane_equations: list[np.ndarray], 
    target_spacing_meters: float = 0.05, 
    len_buffer: float = 0.9, 
    wall_buffer: float = 0.10
) -> np.ndarray:
    """
    Executes line-of-sight ray casting on the downsampled point cloud using an interpolated continuous trajectory.
    Contains a global post-processing override that prevents any point close to ANY wall 
    from being classified as a ghost point.
    """
    print("Pre-processing: Interpolating continuous scanner trajectory path...")
    dense_trajectory = densify_trajectory(trajectory_points, target_spacing_meters=target_spacing_meters)
    print(f"-> Trajectory expanded to {len(dense_trajectory):,} coordinates.")
    
    num_points = len(down_points)
    ghost_mask = np.zeros(num_points, dtype=bool)
    
    print("Step 1: Mapping downsampled points to nearest trajectory origins via KD-Tree...")
    nearest_waypoints = find_nearest_trajectory_waypoints(down_points, dense_trajectory)
    ray_dirs = down_points - nearest_waypoints
    
    print(f"Step 2: Initial Ray-Casting Pass (Evaluating intersections across {len(wall_panels)} panels)...")
    for idx, panel in enumerate(wall_panels):
        A, B, C, D = all_plane_equations[panel['wall_idx']]
        N = np.array([A, B, C])
        
        denominators = np.dot(ray_dirs, N)
        valid_denoms = np.abs(denominators) > 1e-6
        numerators = -(np.dot(nearest_waypoints, N) + D)
        
        t = np.zeros(num_points)
        t[valid_denoms] = numerators[valid_denoms] / denominators[valid_denoms]
        
        potential_hit_indices = np.where((t > 0.0) & (t < len_buffer))[0]
        
        for p_idx in potential_hit_indices:
            intersection_point = nearest_waypoints[p_idx] + t[p_idx] * ray_dirs[p_idx]
            if is_point_in_rectangle_3d(intersection_point, panel['corners']):
                ghost_mask[p_idx] = True

    print("Step 3: Global Proximity Override (Enforcing absolute wall safety buffers)...")
    ghost_indices = np.where(ghost_mask)[0]
    
    if len(ghost_indices) > 0:
        ghost_points_coords = down_points[ghost_indices]
        
        for panel in wall_panels:
            A, B, C, D = all_plane_equations[panel['wall_idx']]
            plane_norm = np.array([A, B, C])
            
            orthogonal_distances = np.abs(np.dot(ghost_points_coords, plane_norm) + D)
            buffered_points_mask = orthogonal_distances <= wall_buffer
            
            actual_points_to_restore = ghost_indices[buffered_points_mask]
            ghost_mask[actual_points_to_restore] = False

    print(f"-> Pipeline Finished: {np.sum(ghost_mask):,} Total Confirmed Ghosts.")
    return ghost_mask


def propagate_labels_kdtree(
    raw_points: np.ndarray, 
    down_points: np.ndarray, 
    ghost_mask_down: np.ndarray
) -> np.ndarray:
    """
    Executes raw spatial label interpolation using a 1-NN KD-Tree query.

    Args:
        raw_points (np.ndarray): Coordinate matrix of the original full cloud, shape (N, 3).
        down_points (np.ndarray): Coordinate matrix of the downsampled cloud, shape (M, 3).
        ghost_mask_down (np.ndarray): Boolean classification mask from downsampled pass, shape (M,).

    Returns:
        np.ndarray: Initial raw-resolution boolean classification mask of shape (N,).
    """
    print("Executing Step 1: Building spatial index and interpolating labels...")
    
    # Construct balanced binary tree from downsampled coordinates
    tree = cKDTree(down_points)
    
    # Query the tree using all full-resolution coordinates
    _, nearest_indices = tree.query(raw_points, k=1, workers=-1)
    
    # Direct index mapping to transfer the downsampled state to full resolution
    initial_ghost_mask = ghost_mask_down[nearest_indices]
    
    print(f"-> Interpolation complete. Seeded {np.sum(initial_ghost_mask):,} potential ghost points.")
    return initial_ghost_mask

def refine_ghost_classifications(
    raw_points: np.ndarray, 
    raw_intensities: np.ndarray, 
    initial_ghost_mask: np.ndarray, 
    wall_panels: list[dict], 
    all_plane_equations: list[np.ndarray], 
    intensity_threshold: float = 0.5, 
    wall_buffer: float = 0.10
) -> np.ndarray:
    """
    Steps 2 & 3: Prunes candidate ghost points using a cascading filter hierarchy.
    Applies the element-wise scalar intensity check first, followed by the floating-point
    matrix dot-product wall proximity calculation.

    Args:
        raw_points (np.ndarray): Coordinate matrix of the original full cloud, shape (N, 3).
        raw_intensities (np.ndarray): Scalar reflection intensity array, shape (N,).
        initial_ghost_mask (np.ndarray): The unrefined boolean mask generated by KD-Tree pass, shape (N,).
        wall_panels (list[dict]): List of bounding structural wall panel dictionaries.
        all_plane_equations (list[np.ndarray]): List of normalized [A, B, C, D] plane coefficients.
        intensity_threshold (float, optional): Scalar threshold above which a point is clean. Defaults to 0.5.
        wall_buffer (float, optional): Absolute perpendicular safety envelope in meters. Defaults to 0.10.

    Returns:
        np.ndarray: Final refined boolean mask of shape (N,).
    """
    # Create an independent array copy to preserve the initial interpolation state in memory
    final_ghost_mask = np.copy(initial_ghost_mask)
    
    # Intensity Filter Sieve
    ghost_indices = np.where(final_ghost_mask)[0]
    if len(ghost_indices) == 0:
        return final_ghost_mask
        
    print(f"Executing Step 2: Evaluating intensity threshold ({intensity_threshold}) across candidates...")
    
    # Identify ghost candidates whose material return matches or exceeds the threshold
    high_intensity_ghost_mask = raw_intensities[ghost_indices] >= intensity_threshold
    intensity_cleared_indices = ghost_indices[high_intensity_ghost_mask]
    
    # Override classification state back to Clean (False)
    final_ghost_mask[intensity_cleared_indices] = False
    
    intensity_dropped_count = len(intensity_cleared_indices)
    print(f"-> Intensity sieve complete. Protected {intensity_dropped_count:,} highly reflective points.")
    
    # Wall Distance Filter Sieve (Matrix/Vector Operations)
    ghost_indices = np.where(final_ghost_mask)[0]
    if len(ghost_indices) == 0:
        return final_ghost_mask
        
    print(f"Executing Step 3: Computing absolute proximity to {len(wall_panels)} wall planes...")
    ghost_coords = raw_points[ghost_indices]
    
    for panel in wall_panels:
        A, B, C, D = all_plane_equations[panel['wall_idx']]
        plane_norm = np.array([A, B, C])
        
        # Shortest perpendicular distance formula to infinite structural plane: d = |Ax + By + Cz + D|
        orthogonal_distances = np.abs(np.dot(ghost_coords, plane_norm) + D)
        
        # Mask matching coordinates residing within the strict spatial safety envelope
        near_wall_mask = orthogonal_distances <= wall_buffer
        
        # Map back to master array indices and override classification state to Clean (False)
        wall_cleared_indices = ghost_indices[near_wall_mask]
        final_ghost_mask[wall_cleared_indices] = False
        
        # Dynamically shrink the remaining computation target pool within the loop
        still_ghost_mask = ~near_wall_mask
        ghost_indices = ghost_indices[still_ghost_mask]
        ghost_coords = ghost_coords[still_ghost_mask]
        
        if len(ghost_indices) == 0:
            break
            
    wall_dropped_count = len(initial_ghost_mask[initial_ghost_mask]) - intensity_dropped_count - len(ghost_indices)
    print(f"-> Wall proximity sieve complete. Protected {wall_dropped_count:,} points near structures.")
    print(f"-> Pipeline Finished: {np.sum(final_ghost_mask):,} Total Confirmed Ghosts | {np.sum(~final_ghost_mask):,} Total Confirmed Clean.")
    
    return final_ghost_mask