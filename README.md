# Reflectance Processing for Interior LiDAR Point Clouds

![DOI]*(Placeholder)*

## Project Description
This repository will contain a Python-based processing pipeline designed to automate the detection and removal of mirror-induced "ghost points" in indoor mobile mapping LiDAR data. Instead of manually clipping out bad data from point clouds, this workflow will ingest a point cloud and trajectory path to programatically idetify mirrors and exclude visual artifacts.

## Data Description
* **LiDAR Point Clouds:** Raw unstructured point clouds exported as `.las`/`.laz` (or `.e57`) files containing $(X, Y, Z)$ coordinates, intensity, and surface normals.
* **Trajectory Data:** Time-stamped spatial trajectory files ($X, Y, Z$) representing the exact path of the mobile scanner during capture.
