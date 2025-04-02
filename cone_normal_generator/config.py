"""
Constants for the Cone Normal Map Generator.
"""

# Default values
DEFAULT_SIZE = 512
DEFAULT_HEIGHT = 1.0
DEFAULT_STRENGTH = 5.0
DEFAULT_DIAMETER = 80  # as percentage of image size (previously DEFAULT_RADIUS = 40)
DEFAULT_MATCAP_ROTATION = 0  # degrees
DEFAULT_SEGMENTS = 1  # Default to standard cone (no segments)
DEFAULT_SEGMENT_RATIO = 50  # Default to equal out-in ratio (50% out, 50% in)

# Preview scaling factors
FAST_PREVIEW_SCALE = 0.25  # Scale factor for fast preview (lower = faster)
DRAG_PREVIEW_SCALE = 0.15  # Even smaller scale during slider dragging

# Timing config
AUTO_REFRESH_DELAY = 300  # Delay in ms before auto-refreshing to avoid too frequent updates
SLIDER_DRAG_DELAY = 500  # Longer delay during slider dragging for better performance

# Folder paths
TEMP_FOLDER = "temp"
OUTPUT_FOLDER = "output"
ASSETS_FOLDER = "assets" 