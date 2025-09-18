# Structure Tensor Analysis Toolkit
=================================

This project provides a set of Python tools to perform 3D Structure Tensor (ST) analysis on volumetric microscopy data.
It allows estimation of orientation vectors, local disarray, and fractional anisotropy from 3D image stacks, with options for visualization and mesh integration.

Repository Structure
--------------------
- st_analysis.py – Main script for structure tensor–based orientation analysis.
- plot_vectors_on_frames.py – Visualization of orientation vectors over image frames.
- 8_insert_real_fibres.py – Insert real fiber orientations into meshes and generate .lon, .vec, .vpts files.
- meshIO.py – Reading and writing of mesh formats (.pts, .elem, .lon, .cpts).
- custom_tool_kit.py, custom_image_base_tool.py – Utility modules for numerical analysis, image handling, and plotting.

Requirements
------------
- Python >= 3.8
- Recommended libraries:
  pip install numpy scipy matplotlib scikit-image tifffile pillow


Usage
-----

1. Orientation Analysis
   Run the main script on a 3D TIFF stack with a parameter file:
   python st_analysis.py -s path/to/stack.tif -p parameters.txt

   This will:
   - Perform structure tensor orientation analysis,
   - Save results in an R_*.npy file (orientation matrix),
   - Write analysis info into Orientation_INFO_*.txt.


2. Visualization
   Overlay orientation vectors on image frames:
   python plot_vectors_on_frames.py

   (Default parameters can be edited inside the script.)

3. Insert Real Fibres into Mesh
   Map orientation vectors onto a mesh:
   python 8_insert_real_fibres.py -msh path/to/mesh -r path/to/R.npy -v

   Generates .lon, .vec, and .vpts files for visualization.

Input Files
-----------
- 3D image stack (TIFF or folder of TIFFs).
- parameters.txt containing acquisition settings (pixel sizes, ROI size, FWHM, thresholds).

Output
------
- R_*.npy – Orientation matrix containing eigenvalues/vectors and shape parameters.
- .txt reports with analysis statistics.
- Optional histograms, TIFF maps, and mesh .lon files.



