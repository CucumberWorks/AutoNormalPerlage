# Cone Normal Map Generator

This application generates a cone-shaped normal map that can be used in 3D graphics applications, game development, or digital art.

## Features

- Creates a height map with a cone shape
- Converts the height map into a normal map
- Adjustable parameters for size, height, strength, and radius
- Outputs both the height map and normal map as PNG files
- Interactive GUI with a dark theme for real-time parameter adjustments
- Previews of both height map and normal map in the application
- Organized file storage in dedicated folders

## Requirements

- Python 3.x
- NumPy
- Pillow (PIL)
- SciPy
- Tkinter (included with most Python installations)

## Installation

1. Clone this repository or download the script files
2. Install the required dependencies:

```bash
# Installing directly to system Python
pip install numpy pillow scipy

# OR create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pillow scipy
```

## Usage

Run the application using Python:

```bash
python cone_normal_map_tk.py
```

The GUI allows you to:
- Select image size (128, 256, 512, 1024, or 2048 pixels)
- Adjust cone radius as a percentage of the image size
- Modify cone height
- Change normal map strength
- Generate and save the images with a single click
- Open the output folder directly from the application
- Clean the output folder when needed

## File Organization

The application organizes files into dedicated folders:

- `temp/`: Contains temporary files used for preview in the application
- `output/`: Contains all generated and saved image files

When the application closes, it automatically cleans up temporary files in the temp folder.

## Output

The application generates two files in the output folder during generation:
- `output/height_map_current.png`: A grayscale image representing the height map
- `output/normal_map_current.png`: An RGB image representing the normal map

When saving, you can choose your own output location and filename prefix. By default, files are saved to the output folder:
- `output/[prefix]_height_map.png`: The saved height map
- `output/[prefix]_normal_map.png`: The saved normal map

## How Normal Maps Work

Normal maps are used in 3D rendering to add surface detail without increasing geometric complexity. Each pixel in a normal map stores a normal vector (X, Y, Z) encoded as RGB colors:

- Red channel: X component of the normal vector
- Green channel: Y component of the normal vector
- Blue channel: Z component of the normal vector

The resulting normal map can be used in various 3D applications to create the illusion of a cone-shaped surface when applied to a flat mesh.

## Dark Mode Interface

This application features a custom dark mode interface for comfortable use in low-light environments and to match modern design standards. The dark theme is designed to be visually appealing and reduce eye strain during extended use. 