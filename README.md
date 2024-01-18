# VoxelOptix
###### Voxel engine using OptiX and some procedural stuff
## Purpose
The main idea of creating this engine was to build some voxel engine using OptiX SDK and CMake with vcpkg. I just wanted to learn how to use CMake properly and have some fun with voxels, so the result isn't optimized at all. During implementation I found (and read about it) that using hardware-accelerated ray tracing doesn't perform well with voxels and procedurally generated geometry in general, so I didn't even try to optimize the result. In terms of performance, there are some better techniques like building a mesh from voxels or tracing rays in a fragment shader. 
## Features
* OpenGL image and text display layer
* Built-in pixel-style font
* Simple user menu
* First person camera
* OptiX AI Denoiser
* Modifiable world generation and render distance
* Ray-marched volumetric clouds
* Procedurally-generated world
* Separate threads for background world/acceleration structures updates
## Screenshots
![image](https://github.com/wm1511/VoxelOptix/assets/72276813/a2e33b14-3d11-4b19-8fc9-64e4c818d485)
![image](https://github.com/wm1511/VoxelOptix/assets/72276813/6cef9c96-c994-4754-bb63-7c5ff345d187)
