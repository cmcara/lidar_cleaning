from pathlib import Path
import numpy as np
import pye57

def load_e57_scan(e57_path: str, scan_index: int = 0):
    # Initialize E57 reader 
    try:
        # Test if the underlying C++ extension initializes safely
        print("pye57 imported successfully!")
    except Exception as e:
        print(f"Error: {e}")

    # Read file
    e57 = pye57.E57(e57_path)

    # Read scan 0 with strict missing-field tolerances
    ## Uses parameter suggested by previous ValueError
    data = e57.read_scan(scan_index, intensity=True, colors=True, ignore_missing_fields=True)

    # Check which fields successfully loaded
    print("Successfully extracted fields from your E57:")
    for key in data.keys():
        print(f" - {key}")

    # Extract raw coordinates
    x = data["cartesianX"]
    y = data["cartesianY"]
    z = data["cartesianZ"]

    # Combine into an (N, 3) matrix
    points_3d = np.column_stack((x, y, z))
    print(f"Spatial matrix shape: {points_3d.shape}")
    print(f"Total points ready for processing: {points_3d.shape[0]:,}")

    # Grab the scanner's transformation matrix
    scanner_pose = e57.scan_position(scan_index)
    print("\nScanner Position Matrix (Pose):")
    print(scanner_pose)

    return points_3d, data, scanner_pose


def inspect_e57_structure(e57_path: str | Path):
    """
    Prints top-level node names, node types, and image keyframe counts from an E57 file.

    Args:
        e57_path (str | Path): Path to the .e57 file.

    Returns:
        None
    """
    e57_file = pye57.E57(str(e57_path))
    root_node = e57_file.image_file.root()

    print("--- Root Level Nodes Available ---")
    for i in range(len(root_node)):
        child = root_node[i]
        print(f"Node Name: {child.elementName()} | Type: {type(child).__name__}")

    try:
        images_node = root_node["images2D"]
        print(f"Found {len(images_node)} trajectory keyframes inside images2D.\n")
    except Exception:
        print("No 'images2D' node found in root structure.\n")


def extract_trajectory(e57_path: str | Path, verbose: bool = False) -> np.ndarray:
    """
    Extracts scanner trajectory waypoints (3D translation positions) from an E57 file.

    Args:
        e57_path (str | Path): Path to the .e57 file.
        verbose (bool, optional): If True, prints individual waypoint coordinates. Defaults to False.

    Returns:
        np.ndarray: NumPy array of shape (N, 3) containing [X, Y, Z] scanner positions.
    """
    e57_file = pye57.E57(str(e57_path))
    root_node = e57_file.image_file.root()

    try:
        images_node = root_node["images2D"]
    except Exception:
        print("No 'images2D' node found in E57 root structure.")
        return np.empty((0, 3))

    trajectory_points = []

    for i in range(len(images_node)):
        img = images_node[i]
        try:
            pose = img["pose"]
            trans = pose["translation"]

            tx = trans["x"].value()
            ty = trans["y"].value()
            tz = trans["z"].value()

            trajectory_points.append([tx, ty, tz])
            if verbose:
                print(f"Waypoint #{i}: Scanner Position = [{tx:.3f}, {ty:.3f}, {tz:.3f}]")

        except Exception as e:
            if verbose:
                print(f"Could not parse Waypoint #{i}. Error signature: {str(e)}")

    trajectory = np.array(trajectory_points)
    print(f"Successfully compiled trajectory matrix with shape: {trajectory.shape}")
    return trajectory