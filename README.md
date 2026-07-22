# Open-Source Automated Reflectance Processing for Interior LiDAR Point Clouds *(In Testing)*

![DOI]  *(Placeholder)*

## Project Description
> This repository contains a Python-based processing workflow designed to test the parameters for a process to automate the detection and removal of mirror-induced "ghost points" in indoor mobile LiDAR scanning data. Current industry-standard workflows involve manually clipping and sifting through point clouds for bad data and could be significantly expedited through automation. This workflow is currently in the testing and fine-tuning stage as I evaluate algorithm performance on a significant number of point clouds.

## Data Description
> **LiDAR Point Clouds:** [`.e57`](http://www.libe57.org/) files containing $(X, Y, Z)$ point cloud coordinates with intensity and separate scanner locations for each time a panorama photograph is taken during the scan. Unlike a static scan where each point in the cloud came from a fixed source, mobile laser scanning involves capturing the same points repeatedly and keeping the best measurements, which presents unique challenges for this workflow.

## Project Status & Execution Instructions
> The core logic has been established and is currently being tested on various interior point cloud environments. The classification algorithms are sensitive to parameter tuning, and finding the best settings may require extensive testing prior to deployment. 

## Core Dependencies
>* ['pye57'](https://github.com/davidcaron/pye57) v0.4.19 developed by [David Caron](https://github.com/davidcaron) allows for reading and writing to .e57 files from Python instead of C++, which is the native language for the file format.
>* [RANSAC](https://www.mathworks.com/discovery/ransac.html) is the foundational algorithm for plane detection, and the Python implementation, along with a number of helpful tools used here, was developed by [Open3D](https://github.com/isl-org/open3d).
>* [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Plotly](https://plotly.com/python/), and [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/).