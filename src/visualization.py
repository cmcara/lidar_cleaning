"""
This is the src file containing all of my plotting and visualization functions for the workflow.
With such spatially sensitive data, I found it very helpful to visualize almost every major step as the data was classified.
If/else statements are core to plotting so much data and allow for display limits to be easily set for various scenarios.
A number of the visualization functions have overlapping features, but I found it more straitforward to add new elements to new functions.
Ideally there would be one function to plot plane classification with a toggle for 'walls vs floors' and 'all planes separately colored'.
"""


import random
import numpy as np
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider


def plot_downsampled_3d(
    pcd_down,
    max_display_points: int = 100000,
    size: float = 1.5,
    opacity: float = 0.8,
):
    """
    Plots a 3D scatter plot of a downsampled point cloud colored by Z-height.

    Args:
        pcd_down (o3d.geometry.PointCloud or np.ndarray): Open3D point cloud or 
            NumPy array of shape (N, 3) containing coordinates.
        max_display_points (int, optional): Maximum number of points to render. 
            Subsamples if exceeded. Defaults to 100000.
        size (float, optional): Point size in the scatter plot. Defaults to 1.5.
        opacity (float, optional): Point opacity from 0.0 to 1.0. Defaults to 0.8.

    Returns:
        None
    """
    # Convert the Open3D downsampled cloud back to a NumPy array
    if hasattr(pcd_down, "points"):
        down_points = np.asarray(pcd_down.points)
    else:
        down_points = np.asarray(pcd_down)

    # Slice a subset of points for plotting
    if len(down_points) > max_display_points:
        indices = np.random.choice(len(down_points), max_display_points, replace=False)
        display_points = down_points[indices]
    else:
        display_points = down_points

    # Create 3D Scatter Plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=display_points[:, 0],
                y=display_points[:, 1],
                z=display_points[:, 2],
                mode="markers",
                marker=dict(
                    size=size,
                    color=display_points[:, 2],  # Color by height (Z-axis)
                    colorscale="Viridis",
                    opacity=opacity,
                ),
            )
        ]
    )

    # Set up layout
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",  # Keeps the 1:1:1 geometric proportions intact
        ),
    )

    # Display
    fig.show()

def visualize_fixed_top_down_intensity_slice(
    z_center: float,
    raw_points: np.ndarray,
    raw_intensities: np.ndarray,
    slice_thickness: float = 0.25,
    max_slice_points: int = 300000,
    intensity_min: float | None = None,
    intensity_max: float | None = None,
):
    """
    Renders a fixed-viewport top-down 2D floor plan slice colored by laser return intensity using high-performance WebGL.
    Includes precise coordinate and intensity readouts on hover.
    """
    # Compute global, unchangeable boundaries across the dataset
    x_min, x_max = float(np.min(raw_points[:, 0])), float(np.max(raw_points[:, 0]))
    y_min, y_max = float(np.min(raw_points[:, 1])), float(np.max(raw_points[:, 1]))

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    locked_x_range = [x_min - x_pad, x_max + x_pad]
    locked_y_range = [y_min - y_pad, y_max + y_pad]

    # Fallback bounds if not explicitly passed by wrapper
    if intensity_min is None:
        intensity_min = float(np.min(raw_intensities))
    if intensity_max is None:
        intensity_max = float(np.max(raw_intensities))

    lower_bound = z_center - (slice_thickness / 2.0)
    upper_bound = z_center + (slice_thickness / 2.0)

    slice_condition = (raw_points[:, 2] >= lower_bound) & (raw_points[:, 2] <= upper_bound)
    slice_indices = np.where(slice_condition)[0]

    total_in_slice = len(slice_indices)
    if total_in_slice == 0:
        print(f"No points found between {lower_bound:.2f}m and {upper_bound:.2f}m")
        return

    if total_in_slice > max_slice_points:
        sampled_indices = np.random.choice(slice_indices, max_slice_points, replace=False)
        slice_coords = raw_points[sampled_indices]
        slice_intensities = raw_intensities[sampled_indices]
        display_text = f"Showing {max_slice_points:,} points (Randomly sampled from {total_in_slice:,})"
    else:
        slice_coords = raw_points[slice_indices]
        slice_intensities = raw_intensities[slice_indices]
        display_text = f"Showing all {total_in_slice:,} points in this vertical slice"

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=slice_coords[:, 0],
        y=slice_coords[:, 1],
        mode='markers',
        text=slice_intensities,
        hovertemplate=(
            "<b>Point Coordinates</b><br>" +
            "X: %{x:.3f} m<br>" +
            "Y: %{y:.3f} m<br>" +
            "Intensity: %{text:.4f}<br>" +
            "<extra></extra>"
        ),
        marker=dict(
            size=2.0,
            color=slice_intensities,
            colorscale='Viridis',
            cmin=intensity_min,
            cmax=intensity_max,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Raw Intensity",
                    side="top"
                ),
                ticks="outside"
            )
        )
    ))

    fig.update_layout(
        title=dict(
            text=f"Diagnostic Intensity Slice: Z = {z_center:.2f}m (± {slice_thickness*100/2:.1f}cm)<br><sub>{display_text}</sub>"
        ),
        xaxis=dict(
            title="Global X (meters)",
            range=locked_x_range,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title="Global Y (meters)",
            range=locked_y_range
        ),
        width=1000,
        height=800,
        template="plotly_white",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="monospace"
        )
    )
    fig.show(config={"displayModeBar": True})


def plot_interactive_top_down_intensity_slice(
    raw_points: np.ndarray,
    raw_intensities: np.ndarray,
    slice_thickness: float = 0.25,
    step: float | None = None,
):
    """
    Calculates Z and intensity bounds automatically and launches the interactive top-down intensity slider widget.
    If 'step' is not provided, it defaults to 'slice_thickness' for gapless slicing.
    """
    if step is None:
        step = slice_thickness

    z_min = float(np.min(raw_points[:, 2]))
    z_max = float(np.max(raw_points[:, 2]))

    # Lock colorbar bounds globally across all slices
    intensity_min = float(np.min(raw_intensities))
    intensity_max = float(np.max(raw_intensities))

    print("--- Diagnostic Interactive Framework Initialized ---")
    print("Hover over any point to read its precise coordinates and intensity value.")

    interact(
        lambda z: visualize_fixed_top_down_intensity_slice(
            z_center=z,
            raw_points=raw_points,
            raw_intensities=raw_intensities,
            slice_thickness=slice_thickness,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
        ),
        z=FloatSlider(
            min=z_min,
            max=z_max,
            step=step,
            value=round((z_min + z_max) / 2.0, 2),
            description="Z-Height (m)",
            layout={'width': '600px'},
            continuous_update=False,
        ),
    )

def plot_classified_planes(
    plane_clouds: list[np.ndarray],
    plane_equations: list[np.ndarray],
    remaining_pcd,
    max_plane_pts: int = 5000,
    max_clutter_pts: int = 35000,
    plane_pt_size: float = 1.5,
    clutter_pt_size: float = 1.0,
    plane_opacity: float = 0.85,
    clutter_opacity: float = 0.5,
):
    """
    Plots classified structural planes (horizontal slabs vs vertical walls) 
    along with unsegmented clutter using 3D Plotly scatter traces.

    Args:
        plane_clouds (list[np.ndarray]): List of point arrays for each extracted plane.
        plane_equations (list[np.ndarray]): List of plane equations [A, B, C, D].
        remaining_pcd (o3d.geometry.PointCloud or np.ndarray): Remaining unsegmented points.
        max_plane_pts (int, optional): Maximum display points per individual plane. Defaults to 5000.
        max_clutter_pts (int, optional): Maximum display points for remaining clutter. Defaults to 35000.
        plane_pt_size (float, optional): Point marker size for planes. Defaults to 1.5.
        clutter_pt_size (float, optional): Point marker size for clutter. Defaults to 1.0.
        plane_opacity (float, optional): Opacity for plane points (0.0 to 1.0). Defaults to 0.85.
        clutter_opacity (float, optional): Opacity for clutter points (0.0 to 1.0). Defaults to 0.5.

    Returns:
        None
    """
    def sample_array(arr, max_pts):
        if len(arr) > max_pts:
            return arr[np.random.choice(len(arr), max_pts, replace=False)]
        return arr

    fig = go.Figure()

    # Iterate through all extracted planes and color-code by orientation
    for idx, (plane_pts, plane_eq) in enumerate(zip(plane_clouds, plane_equations)):
        disp_plane = sample_array(plane_pts, max_pts=max_plane_pts)
        
        A, B, C, D = plane_eq
        is_horizontal = abs(C) > 0.80
        
        if is_horizontal:
            color_string = 'rgb(44, 122, 123)'  # Deep Teal for Floors/Ceilings
            legend_group = 'Horizontal Slabs'
            trace_name = f'Slab #{idx+1}'
        else:
            color_string = 'rgb(229, 62, 62)'   # Bright Crimson Red for Walls
            legend_group = 'Vertical Walls'
            trace_name = f'Wall #{idx+1}'
        
        fig.add_trace(go.Scatter3d(
            x=disp_plane[:, 0], y=disp_plane[:, 1], z=disp_plane[:, 2],
            mode='markers',
            marker=dict(size=plane_pt_size, color=color_string, opacity=plane_opacity),
            name=trace_name,
            legendgroup=legend_group
        ))

    # Extract and add remaining unsegmented points (noise/clutter)
    if hasattr(remaining_pcd, "points"):
        remaining_pts = np.asarray(remaining_pcd.points)
    else:
        remaining_pts = np.asarray(remaining_pcd)

    disp_remaining = sample_array(remaining_pts, max_pts=max_clutter_pts)

    if len(disp_remaining) > 0:
        fig.add_trace(go.Scatter3d(
            x=disp_remaining[:, 0], y=disp_remaining[:, 1], z=disp_remaining[:, 2],
            mode='markers',
            marker=dict(size=clutter_pt_size, color='rgb(160, 174, 192)', opacity=clutter_opacity),
            name='Unsegmented Clutter/Furniture'
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            aspectmode='data',
            xaxis=dict(backgroundcolor="rgb(247, 250, 252)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(247, 250, 252)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(247, 250, 252)", gridcolor="white")
        ),
        showlegend=True
    )

    fig.show(config={"displayModeBar": False})



def plot_individual_planes(
    plane_clouds: list[np.ndarray],
    remaining_pcd,
    max_plane_pts: int = 5000,
    max_clutter_pts: int = 20000,
    plane_pt_size: float = 1.5,
    clutter_pt_size: float = 1.0,
    plane_opacity: float = 0.8,
    clutter_opacity: float = 0.3,
):
    """
    Plots each detected plane with a distinct random color alongside unsegmented clutter.

    Args:
        plane_clouds (list[np.ndarray]): List of point arrays for each extracted plane.
        remaining_pcd (o3d.geometry.PointCloud or np.ndarray): Remaining unsegmented points.
        max_plane_pts (int, optional): Maximum display points per individual plane. Defaults to 5000.
        max_clutter_pts (int, optional): Maximum display points for remaining clutter. Defaults to 20000.
        plane_pt_size (float, optional): Point marker size for planes. Defaults to 1.5.
        clutter_pt_size (float, optional): Point marker size for clutter. Defaults to 1.0.
        plane_opacity (float, optional): Opacity for plane points (0.0 to 1.0). Defaults to 0.8.
        clutter_opacity (float, optional): Opacity for clutter points (0.0 to 1.0). Defaults to 0.3.

    Returns:
        None
    """
    def sample_array(arr, max_pts):
        if len(arr) > max_pts:
            return arr[np.random.choice(len(arr), max_pts, replace=False)]
        return arr

    fig = go.Figure()

    # Add the discovered planes to the visualizer
    for idx, plane_pts in enumerate(plane_clouds):
        # Downsample this individual plane slightly just for display speed
        disp_plane = sample_array(plane_pts, max_pts=max_plane_pts)
        
        # Generate a random bright color for this specific plane layer
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        color_string = f'rgb({r},{g},{b})'
        
        # Add this plane as its own individual "trace" layer
        fig.add_trace(go.Scatter3d(
            x=disp_plane[:, 0], y=disp_plane[:, 1], z=disp_plane[:, 2],
            mode='markers',
            marker=dict(size=plane_pt_size, color=color_string, opacity=plane_opacity),
            name=f'Plane #{idx+1}'
        ))

    # Extract and add remaining unsegmented points (noise/clutter)
    if hasattr(remaining_pcd, "points"):
        remaining_pts = np.asarray(remaining_pcd.points)
    else:
        remaining_pts = np.asarray(remaining_pcd)

    disp_remaining = sample_array(remaining_pts, max_pts=max_clutter_pts)

    if len(disp_remaining) > 0:
        fig.add_trace(go.Scatter3d(
            x=disp_remaining[:, 0], y=disp_remaining[:, 1], z=disp_remaining[:, 2],
            mode='markers',
            marker=dict(size=clutter_pt_size, color='lightgrey', opacity=clutter_opacity),
            name='Unsegmented Clutter'
        ))

    # Maintain true architectural scale (1:1:1)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(aspectmode='data'),
        showlegend=True
    )

    fig.show(config={"displayModeBar": False})



def plot_reconstructed_walls(
    plane_clouds: list[np.ndarray],
    plane_equations: list[np.ndarray],
    wall_rectangles: list[dict],
    doorways: list[np.ndarray],
    remaining_pcd,
    max_plane_pts: int = 5000,
    max_clutter_pts: int = 35000,
    plane_pt_size: float = 1.5,
    clutter_pt_size: float = 1.0,
    door_marker_size: float = 8.0,
    panel_opacity: float = 0.6,
    plane_opacity: float = 0.3,
    clutter_opacity: float = 0.4,
):
    """
    Plots raw structural point clouds alongside reconstructed 3D solid wall mesh 
    panels, detected doorway markers, and remaining unclassified clutter.

    Args:
        plane_clouds (list[np.ndarray]): List of point arrays for each extracted plane.
        plane_equations (list[np.ndarray]): List of plane equations [A, B, C, D].
        wall_rectangles (list[dict]): List of wall panel dictionaries with 'corners' key.
        doorways (list[np.ndarray]): List of 3D coordinates representing doorway centers.
        remaining_pcd (o3d.geometry.PointCloud or np.ndarray): Remaining unsegmented points.
        max_plane_pts (int, optional): Max display points per plane cloud. Defaults to 5000.
        max_clutter_pts (int, optional): Max display points for clutter. Defaults to 35000.
        plane_pt_size (float, optional): Size of plane scatter markers. Defaults to 1.5.
        clutter_pt_size (float, optional): Size of clutter scatter markers. Defaults to 1.0.
        door_marker_size (float, optional): Size of doorway marker diamonds. Defaults to 8.0.
        panel_opacity (float, optional): Opacity of solid wall meshes (0.0 to 1.0). Defaults to 0.6.
        plane_opacity (float, optional): Opacity of plane points (0.0 to 1.0). Defaults to 0.3.
        clutter_opacity (float, optional): Opacity of clutter points (0.0 to 1.0). Defaults to 0.4.

    Returns:
        None
    """
    def sample_array(arr, max_pts):
        if len(arr) > max_pts:
            return arr[np.random.choice(len(arr), max_pts, replace=False)]
        return arr

    fig = go.Figure()

    # Plot raw structure data
    added_wall_legend, added_slab_legend = False, False
    for idx, (plane_pts, plane_eq) in enumerate(zip(plane_clouds, plane_equations)):
        disp_plane = sample_array(plane_pts, max_pts=max_plane_pts)
        is_horizontal = abs(plane_eq[2]) > 0.70
        
        color = 'rgb(44, 122, 123)' if is_horizontal else 'rgb(229, 62, 62)'
        group = 'Horizontal Slabs' if is_horizontal else 'Vertical Walls'
        show_lgd = not added_slab_legend if is_horizontal else not added_wall_legend
        
        if is_horizontal:
            added_slab_legend = True
        else:
            added_wall_legend = True

        fig.add_trace(go.Scatter3d(
            x=disp_plane[:, 0], y=disp_plane[:, 1], z=disp_plane[:, 2],
            mode='markers',
            marker=dict(size=plane_pt_size, color=color, opacity=plane_opacity),
            name=group, legendgroup=group, showlegend=show_lgd
        ))

    # Plot aligned solid wall overlays
    added_overlay_legend = False
    for box in wall_rectangles:
        c = box['corners'] 
        show_lgd = not added_overlay_legend
        added_overlay_legend = True
        
        fig.add_trace(go.Mesh3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            i=[0, 0], j=[1, 2], k=[2, 3], 
            color='rgb(246, 173, 85)',
            opacity=panel_opacity,
            name='Completed Structural Panels',
            legendgroup='Structural Panels',
            showlegend=show_lgd
        ))

    # Plot estimated doors center points
    if len(doorways) > 0:
        doors_matrix = np.array(doorways)
        fig.add_trace(go.Scatter3d(
            x=doors_matrix[:, 0], y=doors_matrix[:, 1], z=doors_matrix[:, 2],
            mode='markers',
            marker=dict(size=door_marker_size, color='rgb(72, 187, 120)', symbol='diamond'),
            name='Isolated Doorways', legendgroup='Doorways'
        ))

    # Plot leftover non-floor, non-wall data 
    if hasattr(remaining_pcd, "points"):
        remaining_pts = np.asarray(remaining_pcd.points)
    else:
        remaining_pts = np.asarray(remaining_pcd)

    disp_remaining = sample_array(remaining_pts, max_pts=max_clutter_pts)
    if len(disp_remaining) > 0:
        fig.add_trace(go.Scatter3d(
            x=disp_remaining[:, 0], y=disp_remaining[:, 1], z=disp_remaining[:, 2],
            mode='markers',
            marker=dict(size=clutter_pt_size, color='rgb(160, 174, 192)', opacity=clutter_opacity),
            name='Unclassified Furniture', legendgroup='Furniture'
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(aspectmode='data'),
        showlegend=True
    )
    
    fig.show(config={"displayModeBar": False})


def plot_trajectory_and_walls(
    trajectory_points: np.ndarray,
    wall_rectangles: list[dict],
):
    """
    Plots the scanner trajectory path along with wireframe outlines of reconstructed wall panels.

    Args:
        trajectory_points (np.ndarray): Array of shape (N, 3) containing scanner waypoints.
        wall_rectangles (list[dict]): List of wall panel dictionaries with 'corners' key.

    Returns:
        None
    """
    fig = go.Figure()

    # Plot the continuous scanner trajectory path
    if len(trajectory_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=trajectory_points[:, 0],
            y=trajectory_points[:, 1],
            z=trajectory_points[:, 2],
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            line=dict(color='blue', width=3),
            name='Scanner Trajectory'
        ))

    # Plot the wireframe outlines of reconstructed wall panels
    for wall in wall_rectangles:
        corners = wall['corners']
        closed_loop = np.vstack([corners, corners[0]])
        fig.add_trace(go.Scatter3d(
            x=closed_loop[:, 0],
            y=closed_loop[:, 1],
            z=closed_loop[:, 2],
            mode='lines',
            line=dict(color='rgba(120, 120, 120, 0.6)', width=2),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (m)'),
            yaxis=dict(title='Y (m)'),
            zaxis=dict(title='Z (m)')
        ),
        title='Structural Gap & Trajectory Proximity Audit',
        margin=dict(r=0, l=0, b=0, t=40)
    )

    fig.show()



def visualize_classification_progress(
    down_points: np.ndarray,
    ghost_mask: np.ndarray,
    trajectory_points: np.ndarray,
    wall_panels: list[dict],
    max_render_pts: int = 150000,
):
    """
    Generates a 3D Plotly visualization mapping clean points, ghost points, 
    the scanner trajectory, and the intersecting wall panels.

    Args:
        down_points (np.ndarray): Downsampled 3D point cloud coordinates (N, 3).
        ghost_mask (np.ndarray): Boolean mask where True indicates a ghost point.
        trajectory_points (np.ndarray): Array of shape (M, 3) containing scanner waypoints.
        wall_panels (list[dict]): List of wall panel dictionaries with 'corners' key.
        max_render_pts (int, optional): Maximum total render points across clean and ghost points. 
            Defaults to 150000.

    Returns:
        None
    """
    fig = go.Figure()
    
    # Separate the points based on the boolean classification mask
    clean_pts = down_points[~ghost_mask]
    ghost_pts = down_points[ghost_mask]
    
    # Proportional sampling helper to prevent browser lag while maintaining density ratios
    total_input_pts = len(down_points)
    sample_rate = min(max_render_pts / total_input_pts, 1.0)
    
    def sample_pts(pts, rate):
        if len(pts) == 0:
            return pts
        n_samples = int(len(pts) * rate)
        indices = np.random.choice(len(pts), n_samples, replace=False)
        return pts[indices]
    
    disp_clean = sample_pts(clean_pts, sample_rate)
    disp_ghost = sample_pts(ghost_pts, sample_rate)
    
    # 1. Plot Validated Clean Points (Soft Grey/Green)
    if len(disp_clean) > 0:
        fig.add_trace(go.Scatter3d(
            x=disp_clean[:, 0], y=disp_clean[:, 1], z=disp_clean[:, 2],
            mode='markers',
            marker=dict(size=1.2, color='rgb(200, 200, 200)', opacity=0.4),
            name=f'Validated Clean ({len(clean_pts):,} pts)'
        ))
        
    # 2. Plot Detected Ghost Points (Bright Red)
    if len(disp_ghost) > 0:
        fig.add_trace(go.Scatter3d(
            x=disp_ghost[:, 0], y=disp_ghost[:, 1], z=disp_ghost[:, 2],
            mode='markers',
            marker=dict(size=1.8, color='rgb(239, 68, 68)', opacity=0.85),
            name=f'Detected Ghosts ({len(ghost_pts):,} pts)'
        ))
        
    # 3. Plot Scanner Trajectory Path (Blue Line + Markers)
    if len(trajectory_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=trajectory_points[:, 0], y=trajectory_points[:, 1], z=trajectory_points[:, 2],
            mode='lines',
            marker=dict(size=4, color='rgb(59, 130, 246)'),
            line=dict(color='rgb(59, 130, 246)', width=3),
            name='Scanner Trajectory'
        ))
        
    # 4. Plot Reconstructed Bounding Wall Panels (Orange Wireframes)
    for wall_idx, wall in enumerate(wall_panels):
        corners = wall['corners']
        # Close the loop around the 4 rectangular corners
        closed_loop = np.vstack([corners, corners[0]])
        fig.add_trace(go.Scatter3d(
            x=closed_loop[:, 0], y=closed_loop[:, 1], z=closed_loop[:, 2],
            mode='lines',
            line=dict(color='rgb(246, 173, 85)', width=2.5),
            name=f'Wall Panel {wall["wall_idx"]}' if wall_idx == 0 else '',
            legendgroup='Structural Walls',
            showlegend=True if wall_idx == 0 else False
        ))
        
    # Enforce standard architectural configurations
    fig.update_layout(
        title='Line-of-Sight Ghost Detection Audit Map',
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (m)'),
            yaxis=dict(title='Y (m)'),
            zaxis=dict(title='Z (m)')
        ),
        margin=dict(r=0, l=0, b=0, t=40),
        showlegend=True
    )
    
    fig.show(config={"displayModeBar": False})


def visualize_fixed_top_down_slice(
    z_center: float,
    raw_points: np.ndarray,
    final_mask: np.ndarray,
    slice_thickness: float = 0.25,
    max_slice_points: int = 300000,
):
    """
    Renders a fixed-viewport top-down 2D floor plan slice using high-performance WebGL.
    Maintains a completely static X/Y bounding box across all Z changes.
    """
    # Compute global, unchangeable boundaries across the dataset
    x_min, x_max = float(np.min(raw_points[:, 0])), float(np.max(raw_points[:, 0]))
    y_min, y_max = float(np.min(raw_points[:, 1])), float(np.max(raw_points[:, 1]))

    # Add a 5% visual padding to the locked viewport so points aren't clipped at the edges
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    locked_x_range = [x_min - x_pad, x_max + x_pad]
    locked_y_range = [y_min - y_pad, y_max + y_pad]

    # Define current horizontal slice window
    lower_bound = z_center - (slice_thickness / 2.0)
    upper_bound = z_center + (slice_thickness / 2.0)
    
    # Isolate points inside this vertical bin
    slice_condition = (raw_points[:, 2] >= lower_bound) & (raw_points[:, 2] <= upper_bound)
    slice_indices = np.where(slice_condition)[0]
    
    total_in_slice = len(slice_indices)
    if total_in_slice == 0:
        print(f"No points found between {lower_bound:.2f}m and {upper_bound:.2f}m")
        return
        
    # Notebook Payload Guard: Downsample the slice if it's too massive for the Jupyter-browser link
    if total_in_slice > max_slice_points:
        sampled_indices = np.random.choice(slice_indices, max_slice_points, replace=False)
        slice_coords = raw_points[sampled_indices]
        slice_labels = final_mask[sampled_indices]
        display_text = f"Showing {max_slice_points:,} points (Randomly sampled from {total_in_slice:,} for notebook stability)"
    else:
        slice_coords = raw_points[slice_indices]
        slice_labels = final_mask[slice_indices]
        display_text = f"Showing all {total_in_slice:,} points in this vertical slice"

    # Separate into Clean and Ghost categories
    clean_pts = slice_coords[~slice_labels]
    ghost_pts = slice_coords[slice_labels]
    
    fig = go.Figure()
    
    # Use Scattergl (WebGL) instead of regular Scatter (SVG) for intense hardware acceleration
    if len(clean_pts) > 0:
        fig.add_trace(go.Scattergl(
            x=clean_pts[:, 0],
            y=clean_pts[:, 1],
            mode='markers',
            marker=dict(size=1.5, color='rgba(200, 200, 200, 0.5)'),
            name='Clean Building Assets'
        ))
        
    if len(ghost_pts) > 0:
        fig.add_trace(go.Scattergl(
            x=ghost_pts[:, 0],
            y=ghost_pts[:, 1],
            mode='markers',
            marker=dict(size=2.0, color='rgba(239, 68, 68, 0.85)'),
            name='Filter-Identified Ghosts'
        ))
        
    fig.update_layout(
        title=dict(
            text=f"Floor Plan Slice: Z = {z_center:.2f}m (± {slice_thickness*100/2:.1f}cm)<br><sub>{display_text}</sub>"
        ),
        xaxis=dict(
            title="Global X (meters)", 
            range=locked_x_range,  # Explicitly fixes the horizontal boundary
            scaleanchor="y",       # Guarantees a 1:1 real-world physical aspect ratio
            scaleratio=1
        ),
        yaxis=dict(
            title="Global Y (meters)",
            range=locked_y_range   # Explicitly fixes the vertical boundary
        ),
        width=950,
        height=800,
        template="plotly_white",
        legend=dict(itemsizing='constant')
    )
    fig.show(config={"displayModeBar": True})


def plot_interactive_top_down_slice(
    raw_points: np.ndarray,
    final_mask: np.ndarray,
    slice_thickness: float = 0.25,
    step: float | None = None,
):
    """
    Calculates Z bounds automatically and launches the interactive top-down slider widget.
    If 'step' is not provided, it defaults to 'slice_thickness' for gapless slicing.
    """
    if step is None:
        step = slice_thickness

    z_min = float(np.min(raw_points[:, 2]))
    z_max = float(np.max(raw_points[:, 2]))

    interact(
        lambda z: visualize_fixed_top_down_slice(
            z_center=z,
            raw_points=raw_points,
            final_mask=final_mask,
            slice_thickness=slice_thickness,
        ),
        z=FloatSlider(
            min=z_min,
            max=z_max,
            step=step,
            value=round((z_min + z_max) / 2.0, 2),
            description="Z-Height (m)",
            layout={'width': '600px'},
            continuous_update=False,
        ),
    )