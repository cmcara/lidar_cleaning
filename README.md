# Reflectance Processing for Interior LiDAR Point Clouds *(In Progress)*

![DOI]  *(Placeholder)*

## Project Description
This repository will contain a Python-based processing pipeline designed to automate the detection and removal of mirror-induced "ghost points" in indoor mobile mapping LiDAR data. Instead of manually clipping out bad data from point clouds, this workflow will ingest a point cloud and trajectory path to programatically idetify mirrors and exclude visual artifacts.

## Data Description
* **LiDAR Point Clouds:** Raw unstructured point clouds exported as `.las`/`.laz` (or `.e57`) files containing $(X, Y, Z)$ coordinates, intensity, and surface normals.
* **Trajectory Data:** Time-stamped spatial trajectory files ($X, Y, Z$) representing the exact path of the mobile scanner during capture.

## Project Status & Execution Instructions
> **Status (June 2026):** Project repository initialized for course tracking. Pipeline logic is currently in the testing phase. Previous LiDAR workflow included for reference.

### Currently working:
* [x] Core directory structure and repository organization.
* [x] Environment dependency definitions (`environment.yml`).
* [x] Introductory steps to ingest and visualize .e57 point clouds.
* [x] Initial classification for wall and ceiling/floor recognition.
* [x] Recognition of doors and extension of wall planes over mirrors and small windows.


### Not yet ready:
* [ ] Trajectory analysis per point for ray-tracing.
* [ ] Filtering window and mirror points.
* [ ] Clean and well organized GitHub layout.

### Running the Workflow (Future Implementation)
Once development begins, execute the following commands to spin up the required Python environment:
```bash
conda env create -f environment.yml
conda activate mirror-filter-env
```

## Data Release
Interior LiDAR Point Cloud Data will be captured and uploaded as structured e57 data usuing mobile and static scanners on mirrors and reflective surfaces.
Releases for public and open-source data will be added as needed. 