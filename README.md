# AutoNormalPerlage

A streamlined tool for creating cone-shaped normal maps used in game development and 3D graphics.

## What It Does

AutoNormalPerlage generates high-quality normal maps for simulating detailed surface textures without adding geometry:

- Creates circular/cone patterns used for watchmaking textures like perlage
- Generates both height maps and normal maps
- Includes matcap previews to visualize the 3D effect
- Perfect for game assets, 3D models, and digital art
- Uses Numba acceleration for fast performance

## Features

- **Simple Controls**: Adjust size, height, strength, and radius with intuitive sliders
- **Real-time Preview**: See changes instantly with matcap visualization
- **Pattern Options**: Create single cones or arranged patterns
- **Customizable**: Control segments, rotation, and other parameters
- **Export Options**: Save high-resolution PNG outputs
- **Optimized Performance**: JIT compilation with Numba for faster processing

## Quick Start

1. Run `python cone_normal_map_generator.py`
2. Adjust parameters using the sliders
3. Use the "Generate" button to create your normal map
4. Preview how it will look in 3D with the matcap viewer
5. Save the result using the "Save" button

## Use in Game Engines

The generated normal maps can be directly imported into:
- Unity
- Unreal Engine
- Godot
- Blender
- Other 3D software supporting normal maps

## Requirements

- Python 3.x
- NumPy, Pillow (PIL), SciPy, and Tkinter
- Numba (for JIT compilation and acceleration)

## Installation

```bash
# Install required packages
pip install numpy pillow scipy numba
```

Clone this repository and run the application:

```bash
python cone_normal_map_generator.py
```
