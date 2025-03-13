import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.ndimage import convolve
import threading
import os
import time
import sys
import shutil
import math

# Constants
DEFAULT_SIZE = 512
DEFAULT_HEIGHT = 1.0
DEFAULT_STRENGTH = 5.0
DEFAULT_DIAMETER = 80  # as percentage of image size (previously DEFAULT_RADIUS = 40)
DEFAULT_MATCAP_ROTATION = 0  # degrees
DEFAULT_SEGMENTS = 1  # Default to standard cone (no segments)
FAST_PREVIEW_SCALE = 0.25  # Scale factor for fast preview (lower = faster)
AUTO_REFRESH_DELAY = 300  # Delay in ms before auto-refreshing to avoid too frequent updates

# Folder paths
TEMP_FOLDER = "temp"
OUTPUT_FOLDER = "output"
ASSETS_FOLDER = "assets"

# Ensure folders exist
def ensure_folders_exist():
    """Create necessary folders if they don't exist."""
    for folder in [TEMP_FOLDER, OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

# Create folders when module is imported
ensure_folders_exist()

class ConeNormalMapGenerator:
    def __init__(self):
        self.size = DEFAULT_SIZE
        self.height = DEFAULT_HEIGHT
        self.strength = DEFAULT_STRENGTH
        self.radius_percent = DEFAULT_DIAMETER  # Keeping variable name for compatibility, but using diameter value
        self.matcap_rotation = DEFAULT_MATCAP_ROTATION
        self.segments = DEFAULT_SEGMENTS
        self.use_fast_preview = True
        
        self.height_map = None
        self.normal_map = None
        self.matcap_preview = None
        
        # Load matcap texture if available
        self.matcap_texture = None
        self.load_matcap()
        
        # Temporary image files for preview
        self.height_map_file = os.path.join(TEMP_FOLDER, "height_map_temp.png")
        self.normal_map_file = os.path.join(TEMP_FOLDER, "normal_map_temp.png")
        self.matcap_preview_file = os.path.join(TEMP_FOLDER, "matcap_preview_temp.png")
        
        # Current output files
        self.current_height_file = os.path.join(OUTPUT_FOLDER, "height_map_current.png")
        self.current_normal_file = os.path.join(OUTPUT_FOLDER, "normal_map_current.png")
        self.current_matcap_file = os.path.join(OUTPUT_FOLDER, "matcap_preview_current.png")
        
        # Status and progress
        self.status = "Ready to generate"
        self.generation_in_progress = False
        
        # For preview images
        self.height_image = None
        self.normal_image = None
        self.matcap_image = None
        
        self.update_timer = None  # For auto-refresh throttling
        
        # Fast preview mode caches
        self.preview_size_cache = None
        self.preview_height_map = None

    def load_matcap(self):
        """Load the matcap texture if available."""
        matcap_path = os.path.join(ASSETS_FOLDER, "silver_matcap.png")
        try:
            if os.path.exists(matcap_path):
                self.matcap_texture = Image.open(matcap_path).convert("RGBA")
                print(f"Loaded matcap texture: {matcap_path}")
            else:
                print(f"Matcap texture not found: {matcap_path}")
                self.matcap_texture = None
        except Exception as e:
            print(f"Error loading matcap texture: {str(e)}")
            self.matcap_texture = None

    def apply_matcap(self, normal_map, rotation_degrees=0, fast_preview=False):
        """Apply matcap texture to normal map for visualization."""
        if self.matcap_texture is None:
            return None
        
        # Scale down for fast preview if needed
        if fast_preview:
            scale = FAST_PREVIEW_SCALE
            preview_size = max(int(normal_map.width * scale), 64)  # Minimum size of 64px
            normal_map_small = normal_map.resize((preview_size, preview_size), Image.LANCZOS)
        else:
            normal_map_small = normal_map
            
        # Convert to numpy arrays for faster processing
        normal_map_array = np.array(normal_map_small)
        
        # Ensure the matcap is in RGBA format
        if self.matcap_texture.mode != 'RGBA':
            matcap = self.matcap_texture.convert('RGBA')
        else:
            matcap = self.matcap_texture
            
        matcap_array = np.array(matcap)
        
        # Get dimensions
        height, width, _ = normal_map_array.shape
        matcap_height, matcap_width, _ = matcap_array.shape
        
        # Create vectors for all pixels at once
        # Extract normal vectors and convert from [0,255] to [-1,1]
        nx = (normal_map_array[:, :, 0] / 255.0) * 2.0 - 1.0
        ny = (normal_map_array[:, :, 1] / 255.0) * 2.0 - 1.0
        nz = (normal_map_array[:, :, 2] / 255.0) * 2.0 - 1.0
        
        # Ensure normals are unit length - this is important for consistent matcap rendering
        # across different resolutions
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx = np.divide(nx, norm, out=np.zeros_like(nx), where=norm!=0)
        ny = np.divide(ny, norm, out=np.zeros_like(ny), where=norm!=0)
        nz = np.divide(nz, norm, out=np.zeros_like(nz), where=norm!=0)
        
        # Convert rotation to radians
        rotation_rad = math.radians(rotation_degrees)
        cos_angle = math.cos(rotation_rad)
        sin_angle = math.sin(rotation_rad)
        
        # Apply rotation to normal vectors using vector operations
        nx_rot = nx * cos_angle - ny * sin_angle
        ny_rot = nx * sin_angle + ny * cos_angle
        
        # Convert to UV coordinates for matcap lookup
        # The UV coordinates should be based solely on the normal vectors,
        # not on the resolution of the normal map
        u = np.clip(((nx_rot + 1.0) * 0.5 * (matcap_width - 1)).astype(int), 0, matcap_width - 1)
        v = np.clip(((ny_rot + 1.0) * 0.5 * (matcap_height - 1)).astype(int), 0, matcap_height - 1)
        
        # Use advanced indexing to lookup matcap colors in one operation
        matcap_result = matcap_array[v, u]
        
        # Scale back up if we used fast preview
        if fast_preview and normal_map.width != preview_size:
            result_img = Image.fromarray(matcap_result)
            result_img = result_img.resize((normal_map.width, normal_map.height), Image.LANCZOS)
            return result_img
        else:
            return Image.fromarray(matcap_result)

    def create_cone_height_map(self, fast_preview=False):
        """Create a height map of a cone shape with optional segmentation."""
        # If in fast preview mode, use a smaller size
        actual_size = self.size
        if fast_preview and self.use_fast_preview:
            actual_size = max(int(self.size * FAST_PREVIEW_SCALE), 128)
            
            # Check if we can reuse the cached preview
            if self.preview_size_cache == (actual_size, self.radius_percent, self.height, self.segments):
                return self.preview_height_map
            
        size = actual_size
        center = (size // 2, size // 2)
        
        # Convert diameter percentage to radius percentage (divide by 2)
        radius_percent = self.radius_percent / 2.0
        radius = int(size * (radius_percent / 100.0))
        
        height = self.height
        segments = self.segments
        
        # Create a grid of coordinates
        y, x = np.ogrid[:size, :size]
        
        # Calculate distance from center for each pixel
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        if segments <= 1:
            # Create the standard cone shape (height decreases linearly with distance)
            height_map = np.maximum(0, height * (1.0 - dist_from_center / radius))
        else:
            # Create a segmented cone with alternating in/out rings
            # Normalize distance to be 0 to 1 within the radius
            normalized_dist = np.clip(dist_from_center / radius, 0, 1)
            
            # Scale the distance based on number of segments (multiply by π × segments)
            scaled_dist = normalized_dist * np.pi * segments
            
            # Use cosine to create alternating rings (cosine oscillates between -1 and 1)
            # We want the center to start with positive height (going out) for odd segments
            # and negative height (going in) for even segments
            oscillation = np.cos(scaled_dist)
            
            # Apply height and ensure we fade to 0 at the edges
            height_map = height * oscillation * (1.0 - normalized_dist)
            
            # Set values outside the radius to 0
            height_map[normalized_dist >= 1.0] = 0
        
        # Cache for fast preview mode
        if fast_preview and self.use_fast_preview:
            self.preview_size_cache = (actual_size, self.radius_percent, self.height, self.segments)
            self.preview_height_map = height_map
        
        return height_map

    def height_map_to_normal_map(self, height_map):
        """Convert a height map to a normal map."""
        # Get the dimensions of the height map
        height, width = height_map.shape
        
        # Scale strength based on resolution for consistent results between different sizes
        # Using DEFAULT_SIZE as the reference size - we need to increase strength for higher resolutions
        resolution_scale = max(width, height) / DEFAULT_SIZE  # Invert the ratio to properly scale up for higher resolutions
        adjusted_strength = self.strength * resolution_scale
        
        # Create an empty normal map
        normal_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate gradients using Sobel operators - Use scipy's optimized operators
        # This is much faster than manual convolution
        dx = convolve(height_map, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * adjusted_strength)
        dy = convolve(height_map, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) * adjusted_strength)
        
        # Calculate normal vectors (vectorized operations for speed)
        z = np.ones_like(dx) * 1.0
        
        # Compute normal length once for all components
        normal_length = np.sqrt(dx**2 + dy**2 + z**2)
        
        # Normalize to get unit vectors (vectorized)
        dx_normalized = dx / normal_length
        dy_normalized = dy / normal_length
        z_normalized = z / normal_length
        
        # Map normal vectors to RGB (vectorized)
        # Remap from [-1, 1] to [0, 255]
        normal_map[..., 0] = (dx_normalized * 0.5 + 0.5) * 255  # Red channel (X)
        normal_map[..., 1] = (dy_normalized * 0.5 + 0.5) * 255  # Green channel (Y)
        normal_map[..., 2] = (z_normalized * 0.5 + 0.5) * 255   # Blue channel (Z)
        
        return normal_map

    def generate_maps(self, update_callback=None):
        """Generate both height map and normal map with optional UI callback."""
        def generate_thread():
            self.generation_in_progress = True
            self.status = "Generating height map..."
            if update_callback: 
                update_callback()
            time.sleep(0.1)  # Let the UI update
            
            try:
                # Create height map (full quality)
                self.height_map = self.create_cone_height_map(fast_preview=False)
                
                self.status = "Converting to normal map..."
                if update_callback: 
                    update_callback()
                time.sleep(0.1)  # Let the UI update
                
                # Convert to normal map
                self.normal_map = self.height_map_to_normal_map(self.height_map)
                
                # Prepare preview images
                self.status = "Preparing preview images..."
                if update_callback: 
                    update_callback()
                
                # Ensure folders exist
                ensure_folders_exist()
                
                if self.height_map is not None:
                    # Normalize height map to 0-255 for visualization
                    height_vis = (self.height_map / np.max(self.height_map) * 255).astype(np.uint8)
                    self.height_image = Image.fromarray(height_vis, mode='L')
                    
                    # Save to temp for preview
                    self.height_image.save(self.height_map_file)
                    
                    # Save current version to output folder
                    self.height_image.save(self.current_height_file)
                
                if self.normal_map is not None:
                    self.normal_image = Image.fromarray(self.normal_map, mode='RGB')
                    
                    # Save to temp for preview
                    self.normal_image.save(self.normal_map_file)
                    
                    # Save current version to output folder
                    self.normal_image.save(self.current_normal_file)
                    
                    # Create matcap preview
                    if self.matcap_texture is not None:
                        self.status = "Applying matcap visualization..."
                        if update_callback: 
                            update_callback()
                        
                        # Full generation should respect the fast preview setting
                        # for the initial matcap preview
                        self.matcap_image = self.apply_matcap(
                            self.normal_image, 
                            self.matcap_rotation,
                            fast_preview=self.use_fast_preview
                        )
                        
                        if self.matcap_image:
                            # Save to temp for preview
                            self.matcap_image.save(self.matcap_preview_file)
                            
                            # Save current version to output folder
                            self.matcap_image.save(self.current_matcap_file)
                
                self.status = "Generation complete"
                
            except Exception as e:
                self.status = f"Error: {str(e)}"
                print(f"Error in generation: {str(e)}")
            
            self.generation_in_progress = False
            if update_callback: 
                update_callback()
        
        # Start generation in background
        thread = threading.Thread(target=generate_thread)
        thread.daemon = True
        thread.start()

    def update_matcap_preview(self, update_callback=None):
        """Update only the matcap preview with current rotation."""
        if self.normal_image is None or self.matcap_texture is None:
            return
            
        def update_thread():
            self.status = "Updating matcap visualization..."
            if update_callback:
                update_callback()
                
            try:
                # Apply matcap with current rotation and user's fast preview setting
                self.matcap_image = self.apply_matcap(
                    self.normal_image, 
                    self.matcap_rotation,
                    fast_preview=self.use_fast_preview  # Use the user preference
                )
                
                if self.matcap_image:
                    # Save to temp for preview
                    self.matcap_image.save(self.matcap_preview_file)
                    
                    # Save current version to output folder
                    self.matcap_image.save(self.current_matcap_file)
                
                self.status = "Matcap updated"
            except Exception as e:
                self.status = f"Error updating matcap: {str(e)}"
                print(f"Error updating matcap: {str(e)}")
                
            if update_callback:
                update_callback()
        
        # Start update in background
        thread = threading.Thread(target=update_thread)
        thread.daemon = True
        thread.start()

    def save_images(self, output_prefix="cone"):
        """Save height map and normal map as final images."""
        if self.height_map is None or self.normal_map is None:
            return None, None
        
        try:
            # Normalize height map to 0-255 for visualization
            height_vis = (self.height_map / np.max(self.height_map) * 255).astype(np.uint8)
            
            # Ensure output folder exists
            ensure_folders_exist()
            
            # Determine if the prefix already includes a path
            if os.path.dirname(output_prefix):
                # Use the provided path
                height_file = f"{output_prefix}_height_map.png"
                normal_file = f"{output_prefix}_normal_map.png"
                matcap_file = f"{output_prefix}_matcap.png"
            else:
                # Add the output folder
                height_file = os.path.join(OUTPUT_FOLDER, f"{output_prefix}_height_map.png")
                normal_file = os.path.join(OUTPUT_FOLDER, f"{output_prefix}_normal_map.png")
                matcap_file = os.path.join(OUTPUT_FOLDER, f"{output_prefix}_matcap.png")
            
            # Create and save height map image
            height_img = Image.fromarray(height_vis, mode='L')
            height_img.save(height_file)
            
            # Create and save normal map image
            normal_img = Image.fromarray(self.normal_map, mode='RGB')
            normal_img.save(normal_file)
            
            # Save status message
            status_msg = f"Images saved: {height_file} and {normal_file}"
            
            # If we have a matcap texture, also save the matcap visualization
            if self.matcap_texture is not None:
                # Generate a high-quality matcap render (non-fast mode)
                matcap_img = self.apply_matcap(normal_img, self.matcap_rotation, fast_preview=False)
                if matcap_img:
                    matcap_img.save(matcap_file)
                    status_msg += f" and {matcap_file}"
            
            self.status = status_msg
            
            return height_file, normal_file
        except Exception as e:
            self.status = f"Error saving: {str(e)}"
            print(f"Error saving images: {str(e)}")
            return None, None

    def quick_update_preview(self, update_callback=None):
        """Update the preview based on current parameters without regenerating the full height map."""
        if self.height_map is None and self.preview_height_map is None:
            # If we have no existing height map, generate from scratch
            self.generate_maps(update_callback)
            return
            
        def update_thread():
            self.status = "Updating preview..."
            self.generation_in_progress = True
            if update_callback: 
                update_callback()
            
            try:
                # Create a faster, lower-resolution height map for interactive preview
                # Only use fast preview if the user has enabled it
                use_fast = self.use_fast_preview
                fast_height_map = self.create_cone_height_map(fast_preview=use_fast)
                
                # Update the height image from the height map
                if fast_height_map is not None:
                    # Normalize height map to 0-255 for visualization
                    height_vis = (fast_height_map / np.max(np.abs(fast_height_map)) * 127.5 + 127.5).astype(np.uint8)
                    self.height_image = Image.fromarray(height_vis, mode='L')
                    
                    # Save to temp for preview
                    if update_callback: 
                        update_callback()
                    self.height_image.save(self.height_map_file)
                
                # Convert to normal map
                fast_normal_map = self.height_map_to_normal_map(fast_height_map)
                
                # Update images
                if fast_normal_map is not None:
                    self.normal_image = Image.fromarray(fast_normal_map, mode='RGB')
                    
                    # Save to temp for preview
                    if update_callback: 
                        update_callback()
                    self.normal_image.save(self.normal_map_file)
                    
                    # Update matcap preview
                    if self.matcap_texture is not None:
                        self.matcap_image = self.apply_matcap(
                            self.normal_image, 
                            self.matcap_rotation,
                            fast_preview=use_fast
                        )
                        
                        if self.matcap_image:
                            # Save to temp for preview
                            self.matcap_image.save(self.matcap_preview_file)
                
                self.status = "Preview updated"
            except Exception as e:
                self.status = f"Error updating preview: {str(e)}"
                print(f"Error updating preview: {str(e)}")
            
            self.generation_in_progress = False
            if update_callback: 
                update_callback()
        
        # Start update in background
        thread = threading.Thread(target=update_thread)
        thread.daemon = True
        thread.start()


class DarkModeTheme:
    """Dark theme colors for Tkinter."""
    BACKGROUND = "#1a1a1a"
    DARK_BG = "#232323"
    TEXT = "#ffffff"
    DISABLED_TEXT = "#969696"
    ACCENT = "#4296fa"
    BUTTON_BG = "#464646"
    BUTTON_ACTIVE = "#6e6e6e"
    SLIDER_BG = "#323232"
    GREEN_BUTTON = "#0c7832"
    GREEN_BUTTON_ACTIVE = "#1eb450"
    BLUE_BUTTON = "#1e5a96"
    BLUE_BUTTON_ACTIVE = "#3c96d2"
    BORDER = "#3c3c3c"


class ConeNormalMapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cone Normal Map Generator")
        self.root.geometry("1400x800")
        self.root.configure(bg=DarkModeTheme.BACKGROUND)
        
        # Create generator
        self.generator = ConeNormalMapGenerator()
        
        # Configure style for dark mode
        self.setup_dark_theme()
        
        # Create UI
        self.create_ui()
        
        # Preview images
        self.height_preview = None
        self.normal_preview = None
        self.matcap_preview = None
        self.matcap_texture_preview = None
        
        # Update periodically
        self.root.after(100, self.periodic_update)
        
        # When closing the app, clean up temp files
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.update_timer = None

    def setup_dark_theme(self):
        """Setup dark theme for ttk widgets."""
        self.style = ttk.Style()
        
        # Configure theme
        self.style.theme_use('default')  # Start with default
        
        # Configure colors
        self.style.configure('TFrame', background=DarkModeTheme.BACKGROUND)
        self.style.configure('TLabel', background=DarkModeTheme.BACKGROUND, foreground=DarkModeTheme.TEXT)
        self.style.configure('TButton', background=DarkModeTheme.BUTTON_BG, foreground=DarkModeTheme.TEXT)
        self.style.map('TButton', 
                      background=[('active', DarkModeTheme.BUTTON_ACTIVE)],
                      foreground=[('active', DarkModeTheme.TEXT)])
        
        # Scale style
        self.style.configure('TScale', background=DarkModeTheme.BACKGROUND, 
                           troughcolor=DarkModeTheme.SLIDER_BG)
        
        # Combobox style
        self.style.configure('TCombobox', 
                           fieldbackground=DarkModeTheme.DARK_BG,
                           background=DarkModeTheme.BUTTON_BG,
                           foreground=DarkModeTheme.TEXT,
                           arrowcolor=DarkModeTheme.TEXT)
        
        # Map additional states for combobox
        self.style.map('TCombobox',
                     fieldbackground=[('readonly', DarkModeTheme.DARK_BG)],
                     background=[('readonly', DarkModeTheme.BUTTON_BG)],
                     foreground=[('readonly', DarkModeTheme.TEXT)])
        
        # Fix for combobox dropdown text color in dark mode
        self.root.option_add('*TCombobox*Listbox.background', DarkModeTheme.DARK_BG)
        self.root.option_add('*TCombobox*Listbox.foreground', DarkModeTheme.TEXT)
        self.root.option_add('*TCombobox*Listbox.selectBackground', DarkModeTheme.ACCENT)
        self.root.option_add('*TCombobox*Listbox.selectForeground', DarkModeTheme.TEXT)
        
        # Special button styles
        self.style.configure('Generate.TButton', 
                           background=DarkModeTheme.GREEN_BUTTON, 
                           foreground=DarkModeTheme.TEXT)
        self.style.map('Generate.TButton', 
                     background=[('active', DarkModeTheme.GREEN_BUTTON_ACTIVE)])
        
        self.style.configure('Save.TButton', 
                           background=DarkModeTheme.BLUE_BUTTON, 
                           foreground=DarkModeTheme.TEXT)
        self.style.map('Save.TButton', 
                     background=[('active', DarkModeTheme.BLUE_BUTTON_ACTIVE)])
        
        # Also style a "clean" button
        self.style.configure('Clean.TButton', 
                           background="#964B00", 
                           foreground=DarkModeTheme.TEXT)
        self.style.map('Clean.TButton', 
                     background=[('active', "#B25900")])

    def create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left sidebar
        sidebar = ttk.Frame(main_frame, width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Make sidebar fixed width
        sidebar.pack_propagate(False)
        
        # Parameters section
        parameters_frame = ttk.Frame(sidebar)
        parameters_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(parameters_frame, text="Cone Parameters", font=("Arial", 14)).pack(pady=(0, 10))
        ttk.Separator(parameters_frame).pack(fill=tk.X, pady=5)
        
        # Function to validate numeric entries
        def validate_numeric(P, is_int=False):
            if P == "":
                return True
            try:
                if is_int:
                    val = int(P)
                else:
                    val = float(P)
                return True
            except ValueError:
                return False
        
        # Register validation command
        validate_float = self.root.register(lambda P: validate_numeric(P, False))
        validate_int = self.root.register(lambda P: validate_numeric(P, True))
        
        # Size selection
        size_frame = ttk.Frame(parameters_frame)
        size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(size_frame, text="Image Size:").pack(side=tk.LEFT, padx=5)
        
        size_values = ["128", "256", "512", "1024", "2048"]
        self.size_var = tk.StringVar(value=str(DEFAULT_SIZE))
        size_combo = ttk.Combobox(size_frame, textvariable=self.size_var, values=size_values, state="readonly", width=10)
        size_combo.pack(side=tk.LEFT, padx=5)
        size_combo.bind("<<ComboboxSelected>>", self.on_size_change)
        
        # Additional styling for combobox - this helps with some platforms
        size_combo.configure(background=DarkModeTheme.DARK_BG, foreground=DarkModeTheme.TEXT)
        
        # Radius slider with input
        radius_frame = ttk.Frame(parameters_frame)
        radius_frame.pack(fill=tk.X, pady=5)
        ttk.Label(radius_frame, text="Cone Diameter (%):").pack(anchor=tk.W, padx=5)
        
        # Add a frame for the slider and input
        radius_control_frame = ttk.Frame(radius_frame)
        radius_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.radius_var = tk.DoubleVar(value=DEFAULT_DIAMETER)
        radius_slider = ttk.Scale(radius_control_frame, from_=20, to=180, variable=self.radius_var, command=self.on_radius_change)
        radius_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add number input
        radius_entry = ttk.Entry(radius_control_frame, textvariable=self.radius_var, width=5, validate="key", validatecommand=(validate_int, '%P'))
        radius_entry.pack(side=tk.RIGHT, padx=5)
        radius_entry.bind("<Return>", lambda e: self.on_radius_entry_change())
        radius_entry.bind("<FocusOut>", lambda e: self.on_radius_entry_change())
        
        # Height slider with input
        height_frame = ttk.Frame(parameters_frame)
        height_frame.pack(fill=tk.X, pady=5)
        ttk.Label(height_frame, text="Cone Height:").pack(anchor=tk.W, padx=5)
        
        # Add a frame for the slider and input
        height_control_frame = ttk.Frame(height_frame)
        height_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.height_var = tk.DoubleVar(value=DEFAULT_HEIGHT)
        height_slider = ttk.Scale(height_control_frame, from_=0.1, to=2.0, variable=self.height_var, command=self.on_height_change)
        height_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add number input
        height_entry = ttk.Entry(height_control_frame, textvariable=self.height_var, width=5, validate="key", validatecommand=(validate_float, '%P'))
        height_entry.pack(side=tk.RIGHT, padx=5)
        height_entry.bind("<Return>", lambda e: self.on_height_entry_change())
        height_entry.bind("<FocusOut>", lambda e: self.on_height_entry_change())
        
        # Strength slider with input
        strength_frame = ttk.Frame(parameters_frame)
        strength_frame.pack(fill=tk.X, pady=5)
        ttk.Label(strength_frame, text="Normal Map Strength:").pack(anchor=tk.W, padx=5)
        
        # Add a frame for the slider and input
        strength_control_frame = ttk.Frame(strength_frame)
        strength_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.strength_var = tk.DoubleVar(value=DEFAULT_STRENGTH)
        strength_slider = ttk.Scale(strength_control_frame, from_=1.0, to=10.0, variable=self.strength_var, command=self.on_strength_change)
        strength_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add number input
        strength_entry = ttk.Entry(strength_control_frame, textvariable=self.strength_var, width=5, validate="key", validatecommand=(validate_float, '%P'))
        strength_entry.pack(side=tk.RIGHT, padx=5)
        strength_entry.bind("<Return>", lambda e: self.on_strength_entry_change())
        strength_entry.bind("<FocusOut>", lambda e: self.on_strength_entry_change())
        
        # Segments slider with input
        segments_frame = ttk.Frame(parameters_frame)
        segments_frame.pack(fill=tk.X, pady=5)
        ttk.Label(segments_frame, text="Cone Segments:").pack(anchor=tk.W, padx=5)
        
        # Add a frame for the slider and input
        segments_control_frame = ttk.Frame(segments_frame)
        segments_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.segments_var = tk.IntVar(value=DEFAULT_SEGMENTS)
        
        # Create a wrapper function to ensure integer values
        def on_segments_drag(val):
            # Force integer value during slider dragging
            int_val = int(float(val))
            if int_val != self.segments_var.get():
                self.segments_var.set(int_val)
            self.on_segments_change()
            
        segments_slider = ttk.Scale(segments_control_frame, from_=1, to=100, variable=self.segments_var, command=on_segments_drag)
        segments_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add number input
        segments_entry = ttk.Entry(segments_control_frame, textvariable=self.segments_var, width=5, validate="key", validatecommand=(validate_int, '%P'))
        segments_entry.pack(side=tk.RIGHT, padx=5)
        segments_entry.bind("<Return>", lambda e: self.on_segments_entry_change())
        segments_entry.bind("<FocusOut>", lambda e: self.on_segments_entry_change())
        
        # Matcap section
        matcap_frame = ttk.Frame(parameters_frame)
        matcap_frame.pack(fill=tk.X, pady=10)
        ttk.Label(matcap_frame, text="Matcap Visualization", font=("Arial", 12)).pack(pady=(0, 5))
        ttk.Separator(matcap_frame).pack(fill=tk.X, pady=5)
        
        # Matcap rotation slider with input
        rotation_frame = ttk.Frame(matcap_frame)
        rotation_frame.pack(fill=tk.X, pady=5)
        ttk.Label(rotation_frame, text="Matcap Rotation (°):").pack(anchor=tk.W, padx=5)
        
        # Add a frame for the slider and input
        rotation_control_frame = ttk.Frame(rotation_frame)
        rotation_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.rotation_var = tk.DoubleVar(value=DEFAULT_MATCAP_ROTATION)
        rotation_slider = ttk.Scale(rotation_control_frame, from_=0, to=360, variable=self.rotation_var, command=self.on_rotation_change)
        rotation_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add number input
        rotation_entry = ttk.Entry(rotation_control_frame, textvariable=self.rotation_var, width=5, validate="key", validatecommand=(validate_int, '%P'))
        rotation_entry.pack(side=tk.RIGHT, padx=5)
        rotation_entry.bind("<Return>", lambda e: self.on_rotation_entry_change())
        rotation_entry.bind("<FocusOut>", lambda e: self.on_rotation_entry_change())
        
        # After the matcap rotation slider
        # Add fast preview checkbox
        fast_preview_frame = ttk.Frame(matcap_frame)
        fast_preview_frame.pack(fill=tk.X, pady=5)
        
        self.fast_preview_var = tk.BooleanVar(value=True)
        fast_preview_check = ttk.Checkbutton(
            fast_preview_frame, 
            text="Use Fast Preview", 
            variable=self.fast_preview_var,
            command=self.on_fast_preview_change
        )
        fast_preview_check.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Add quality indicator label
        self.preview_quality_label = ttk.Label(
            fast_preview_frame, 
            text="(Fast Mode)", 
            foreground="#4296fa"
        )
        self.preview_quality_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Buttons
        buttons_frame = ttk.Frame(parameters_frame)
        buttons_frame.pack(fill=tk.X, pady=15)
        
        self.generate_button = ttk.Button(buttons_frame, text="Generate", command=self.on_generate, style="Generate.TButton")
        self.generate_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.save_button = ttk.Button(buttons_frame, text="Save Images", command=self.on_save, style="Save.TButton", state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Clean output folder button
        clean_button_frame = ttk.Frame(parameters_frame)
        clean_button_frame.pack(fill=tk.X, pady=5)
        
        self.clean_button = ttk.Button(clean_button_frame, text="Clean Output Folder", command=self.on_clean_output, style="Clean.TButton")
        self.clean_button.pack(pady=5, fill=tk.X)
        
        # Output folder button
        folder_frame = ttk.Frame(parameters_frame)
        folder_frame.pack(fill=tk.X, pady=5)
        
        open_folder_button = ttk.Button(folder_frame, text="Open Output Folder", 
                                      command=lambda: self.open_folder(OUTPUT_FOLDER))
        open_folder_button.pack(pady=5, fill=tk.X)
        
        # Status section
        status_frame = ttk.Frame(sidebar)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready", wraplength=280)
        self.status_label.pack(pady=5, anchor=tk.W)
        
        # Right content for image display in a 2x2 grid
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Set up image preview area - 2x2 grid
        preview_frame = ttk.Frame(content_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a grid for the previews
        preview_grid = ttk.Frame(preview_frame)
        preview_grid.pack(fill=tk.BOTH, expand=True)
        
        # Row 1: Height Map and Normal Map
        height_frame = ttk.Frame(preview_grid)
        height_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        normal_frame = ttk.Frame(preview_grid)
        normal_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Row 2: Matcap Preview and Matcap Texture
        matcap_preview_frame = ttk.Frame(preview_grid)
        matcap_preview_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        matcap_texture_frame = ttk.Frame(preview_grid)
        matcap_texture_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure the grid to expand properly
        preview_grid.columnconfigure(0, weight=1)
        preview_grid.columnconfigure(1, weight=1)
        preview_grid.rowconfigure(0, weight=1)
        preview_grid.rowconfigure(1, weight=1)
        
        # Height map preview
        ttk.Label(height_frame, text="Height Map", font=("Arial", 12)).pack(pady=(0, 5))
        self.height_canvas = tk.Canvas(height_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.height_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Normal map preview
        ttk.Label(normal_frame, text="Normal Map", font=("Arial", 12)).pack(pady=(0, 5))
        self.normal_canvas = tk.Canvas(normal_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.normal_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Matcap preview
        ttk.Label(matcap_preview_frame, text="Matcap Preview", font=("Arial", 12)).pack(pady=(0, 5))
        self.matcap_preview_canvas = tk.Canvas(matcap_preview_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.matcap_preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Matcap texture 
        ttk.Label(matcap_texture_frame, text="Matcap Texture", font=("Arial", 12)).pack(pady=(0, 5))
        self.matcap_texture_canvas = tk.Canvas(matcap_texture_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.matcap_texture_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial setup
        self.display_empty_previews()
    
    def display_empty_previews(self):
        """Display empty preview canvases."""
        # Create empty images
        empty_img = Image.new('RGB', (DEFAULT_SIZE, DEFAULT_SIZE), color=(50, 50, 50))
        empty_photo = ImageTk.PhotoImage(empty_img)
        
        # Save references to avoid garbage collection
        self.empty_preview_img = empty_photo
        
        # Display on canvases - center the images
        for canvas in [self.height_canvas, self.normal_canvas, 
                      self.matcap_preview_canvas, self.matcap_texture_canvas]:
            canvas.delete("all")
            # Get canvas dimensions
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # Use default dimensions if canvas hasn't been rendered yet
            if canvas_width < 10:  # Not properly sized yet
                canvas_width = 300
                canvas_height = 300
                
            # Calculate center position
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            
            # Place image in center
            canvas.create_image(center_x, center_y, anchor=tk.CENTER, image=self.empty_preview_img)
    
    def on_size_change(self, event=None):
        """Handle change in size selection."""
        size = int(self.size_var.get())
        self.generator.size = size
        
        # For resolution changes, we need a full regeneration, not just a fast preview
        # Clear any preview cache to force regeneration
        self.generator.preview_size_cache = None
        self.generator.preview_height_map = None
        
        # Use full generation instead of quick preview for size changes
        if not self.generator.generation_in_progress:
            self.generator.generate_maps(self.update_ui)
    
    def on_radius_change(self, event=None):
        """Handle change in radius slider."""
        radius = round(self.radius_var.get())
        self.generator.radius_percent = radius
        
        # Schedule preview update
        self.schedule_preview_update()
    
    def on_radius_entry_change(self):
        """Handle direct input in diameter entry."""
        try:
            diameter = int(self.radius_var.get())
            # Constrain to valid range
            diameter = max(20, min(180, diameter))
            self.radius_var.set(diameter)
            self.generator.radius_percent = diameter
            
            # Schedule preview update
            self.schedule_preview_update()
        except ValueError:
            # Reset to last valid value
            self.radius_var.set(self.generator.radius_percent)
    
    def on_height_change(self, event=None):
        """Handle change in height slider."""
        height = round(self.height_var.get(), 1)
        self.generator.height = height
        
        # Schedule preview update
        self.schedule_preview_update()
    
    def on_height_entry_change(self):
        """Handle direct input in height entry."""
        try:
            height = float(self.height_var.get())
            # Constrain to valid range
            height = max(0.1, min(2.0, height))
            self.height_var.set(round(height, 1))
            self.generator.height = height
            
            # Schedule preview update
            self.schedule_preview_update()
        except ValueError:
            # Reset to last valid value
            self.height_var.set(self.generator.height)
    
    def on_strength_change(self, event=None):
        """Handle change in strength slider."""
        strength = round(self.strength_var.get(), 1)
        self.generator.strength = strength
        
        # Schedule preview update
        self.schedule_preview_update()
    
    def on_strength_entry_change(self):
        """Handle direct input in strength entry."""
        try:
            strength = float(self.strength_var.get())
            # Constrain to valid range
            strength = max(1.0, min(10.0, strength))
            self.strength_var.set(round(strength, 1))
            self.generator.strength = strength
            
            # Schedule preview update
            self.schedule_preview_update()
        except ValueError:
            # Reset to last valid value
            self.strength_var.set(self.generator.strength)
    
    def on_segments_change(self, event=None):
        """Handle change in segments slider."""
        # Ensure we're using an integer value
        segments = int(self.segments_var.get())
        self.segments_var.set(segments)  # Force integer in the UI
        self.generator.segments = segments
        
        # For segment changes, we need to clear the preview cache
        # to force a full regeneration of the height map
        self.generator.preview_size_cache = None
        self.generator.preview_height_map = None
        
        # Schedule preview update
        self.schedule_preview_update()
    
    def on_segments_entry_change(self):
        """Handle direct input in segments entry."""
        try:
            segments = int(self.segments_var.get())
            # Constrain to valid range
            segments = max(1, min(50, segments))
            self.segments_var.set(segments)
            self.generator.segments = segments
            
            # For segment changes, we need to clear the preview cache
            # to force a full regeneration of the height map
            self.generator.preview_size_cache = None
            self.generator.preview_height_map = None
            
            # Schedule preview update
            self.schedule_preview_update()
        except ValueError:
            # Reset to last valid value
            self.segments_var.set(self.generator.segments)
    
    def on_rotation_change(self, event=None):
        """Handle change in matcap rotation slider."""
        rotation = round(self.rotation_var.get())
        self.generator.matcap_rotation = rotation
        
        # For rotation changes, we only need to update the matcap rendering,
        # not regenerate the normal map or height map.
        if (self.generator.normal_image is not None and 
            self.generator.matcap_texture is not None):
            
            # Direct rendering of rotated matcap without threaded processing
            try:
                # Apply matcap directly with current rotation
                # Use the fast preview setting from the user preference
                matcap_image = self.generator.apply_matcap(
                    self.generator.normal_image, 
                    rotation,
                    fast_preview=self.generator.use_fast_preview
                )
                
                if matcap_image:
                    # Update the matcap image in memory
                    self.generator.matcap_image = matcap_image
                    
                    # Update the matcap preview
                    self.update_matcap_preview()
                    
            except Exception as e:
                print(f"Error updating matcap rotation: {str(e)}")
    
    def on_rotation_entry_change(self):
        """Handle direct input in rotation entry."""
        try:
            rotation = int(self.rotation_var.get())
            # Constrain to valid range (0-360)
            rotation = rotation % 360
            self.rotation_var.set(rotation)
            self.generator.matcap_rotation = rotation
            
            # Update matcap preview
            if (self.generator.normal_image is not None and 
                self.generator.matcap_texture is not None):
                self.on_rotation_change()
        except ValueError:
            # Reset to last valid value
            self.rotation_var.set(self.generator.matcap_rotation)
    
    def on_fast_preview_change(self, event=None):
        """Handle change in fast preview checkbox."""
        use_fast = self.fast_preview_var.get()
        self.generator.use_fast_preview = use_fast
        
        # Update the quality indicator
        if use_fast:
            self.preview_quality_label.config(text="(Fast Mode)", foreground="#4296fa")
        else:
            self.preview_quality_label.config(text="(High Quality)", foreground="#f44336")
        
        # When fast preview is toggled, we should regenerate the preview
        # using the new quality setting
        self.status_label.config(text=f"Status: {'Fast preview mode' if use_fast else 'High quality mode'}")
        
        # Schedule a refresh to update with new quality setting
        self.schedule_preview_update()
    
    def on_generate(self):
        """Handle generate button click."""
        self.generate_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        
        # Start generation with update callback
        self.generator.generate_maps(self.update_ui)
    
    def on_save(self):
        """Handle save button click."""
        if self.generator.height_map is None or self.generator.normal_map is None:
            return
        
        # Ask for file prefix
        initial_dir = os.path.abspath(OUTPUT_FOLDER)
        file_prefix = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            title="Save Normal Map Images",
            filetypes=[("PNG files", "*.png")],
            defaultextension=".png"
        )
        
        if not file_prefix:
            return
        
        # Remove extension if user added one
        file_prefix = os.path.splitext(file_prefix)[0]
        
        # Save images
        height_file, normal_file = self.generator.save_images(file_prefix)
        
        if height_file and normal_file:
            self.status_label.config(text=f"Status: Images saved successfully as {os.path.basename(height_file)} and {os.path.basename(normal_file)}")
    
    def on_clean_output(self):
        """Handle clean output folder button click."""
        if not os.path.exists(OUTPUT_FOLDER):
            return
            
        try:
            # Ask for confirmation
            confirm = messagebox.askyesno(
                "Clean Output Folder", 
                f"Are you sure you want to delete all files in the {OUTPUT_FOLDER} folder?",
                icon=messagebox.WARNING
            )
            
            if confirm:
                # Delete all files in the output folder
                for file_name in os.listdir(OUTPUT_FOLDER):
                    file_path = os.path.join(OUTPUT_FOLDER, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                
                self.status_label.config(text=f"Status: Cleaned output folder")
        except Exception as e:
            self.status_label.config(text=f"Status: Error cleaning output folder: {str(e)}")
    
    def open_folder(self, folder_path):
        """Open the specified folder in the file explorer."""
        try:
            # Make sure the folder exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            # Open the folder using the appropriate method for the OS
            if sys.platform == 'win32':
                os.startfile(os.path.abspath(folder_path))
            elif sys.platform == 'darwin':  # macOS
                os.system(f'open "{os.path.abspath(folder_path)}"')
            else:  # Linux
                os.system(f'xdg-open "{os.path.abspath(folder_path)}"')
        except Exception as e:
            self.status_label.config(text=f"Status: Error opening folder: {str(e)}")
    
    def on_closing(self):
        """Handle window closing event."""
        # Clean up temp folder if it exists
        if os.path.exists(TEMP_FOLDER):
            try:
                # Instead of deleting the folder, just delete the files
                for file_name in os.listdir(TEMP_FOLDER):
                    file_path = os.path.join(TEMP_FOLDER, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up temp folder: {str(e)}")
        
        # Close the window
        self.root.destroy()
    
    def update_ui(self):
        """Update UI with current generator state."""
        # Update status
        self.status_label.config(text=f"Status: {self.generator.status}")
        
        # Enable/disable buttons
        if self.generator.generation_in_progress:
            self.generate_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
        else:
            self.generate_button.config(state=tk.NORMAL)
            if self.generator.height_map is not None and self.generator.normal_map is not None:
                self.save_button.config(state=tk.NORMAL)
        
        # Update previews if available
        if self.generator.height_image is not None:
            self.update_height_preview()
        
        if self.generator.normal_image is not None:
            self.update_normal_preview()
            
        if self.generator.matcap_image is not None:
            self.update_matcap_preview()
            
        # Update matcap texture preview
        self.update_matcap_texture_preview()
    
    def update_height_preview(self):
        """Update height map preview."""
        if self.generator.height_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.height_canvas.winfo_width()
        canvas_height = self.height_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
        
        # Calculate the display size to maintain aspect ratio
        display_size = min(canvas_width, canvas_height)
        
        # Resize image for display
        preview_img = self.generator.height_image.resize((display_size, display_size), Image.LANCZOS)
        photo = ImageTk.PhotoImage(preview_img)
        
        # Keep reference to avoid garbage collection
        self.height_preview = photo
        
        # Calculate center position
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Clear canvas and display centered image
        self.height_canvas.delete("all")
        self.height_canvas.create_image(center_x, center_y, anchor=tk.CENTER, image=self.height_preview)
    
    def update_normal_preview(self):
        """Update normal map preview."""
        if self.generator.normal_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.normal_canvas.winfo_width()
        canvas_height = self.normal_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
        
        # Calculate the display size to maintain aspect ratio
        display_size = min(canvas_width, canvas_height)
        
        # Resize image for display
        preview_img = self.generator.normal_image.resize((display_size, display_size), Image.LANCZOS)
        photo = ImageTk.PhotoImage(preview_img)
        
        # Keep reference to avoid garbage collection
        self.normal_preview = photo
        
        # Calculate center position
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Clear canvas and display centered image
        self.normal_canvas.delete("all")
        self.normal_canvas.create_image(center_x, center_y, anchor=tk.CENTER, image=self.normal_preview)
    
    def update_matcap_preview(self):
        """Update matcap preview."""
        if self.generator.matcap_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.matcap_preview_canvas.winfo_width()
        canvas_height = self.matcap_preview_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
        
        # Calculate the display size to maintain aspect ratio
        display_size = min(canvas_width, canvas_height)
        
        # Resize image for display
        preview_img = self.generator.matcap_image.resize((display_size, display_size), Image.LANCZOS)
        photo = ImageTk.PhotoImage(preview_img)
        
        # Keep reference to avoid garbage collection
        self.matcap_preview = photo
        
        # Calculate center position
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Clear canvas and display centered image
        self.matcap_preview_canvas.delete("all")
        self.matcap_preview_canvas.create_image(center_x, center_y, anchor=tk.CENTER, image=self.matcap_preview)
    
    def update_matcap_texture_preview(self):
        """Update matcap texture preview."""
        if self.generator.matcap_texture is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.matcap_texture_canvas.winfo_width()
        canvas_height = self.matcap_texture_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
        
        # Calculate the display size to maintain aspect ratio
        display_size = min(canvas_width, canvas_height)
        
        # Resize image for display
        preview_img = self.generator.matcap_texture.resize((display_size, display_size), Image.LANCZOS)
        photo = ImageTk.PhotoImage(preview_img)
        
        # Keep reference to avoid garbage collection
        self.matcap_texture_preview = photo
        
        # Calculate center position
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Clear canvas and display centered image
        self.matcap_texture_canvas.delete("all")
        self.matcap_texture_canvas.create_image(center_x, center_y, anchor=tk.CENTER, image=self.matcap_texture_preview)
    
    def periodic_update(self):
        """Periodically update the UI."""
        # Update UI with current generator state
        self.update_ui()
        
        # Reschedule
        self.root.after(100, self.periodic_update)

    def schedule_preview_update(self):
        """Schedule a preview update with a delay to avoid too frequent updates."""
        # Cancel any existing timer
        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        
        # Create a new timer
        self.update_timer = self.root.after(AUTO_REFRESH_DELAY, self.on_delayed_update)
    
    def on_delayed_update(self):
        """Called after delay to update preview."""
        if not self.generator.generation_in_progress:
            self.generator.quick_update_preview(self.update_ui)

def main():
    """Main entry point."""
    print(f"Starting Cone Normal Map Generator (Python {sys.version})")
    
    # Ensure folders exist
    ensure_folders_exist()
    
    root = tk.Tk()
    app = ConeNormalMapApp(root)
    
    # Set window icon if available
    try:
        # Windows and Linux
        icon_path = "normal_map_icon.png"
        if not os.path.exists(icon_path):
            # Create a simple icon
            icon = Image.new("RGB", (64, 64), color="blue")
            # Add some simple normal map like coloring
            pixels = icon.load()
            for i in range(64):
                for j in range(64):
                    x = (i / 64) * 2 - 1
                    y = (j / 64) * 2 - 1
                    dist = min(1, x*x + y*y)
                    z = np.sqrt(1 - dist)
                    r = int((x * 0.5 + 0.5) * 255)
                    g = int((y * 0.5 + 0.5) * 255)
                    b = int((z * 0.5 + 0.5) * 255)
                    pixels[i, j] = (r, g, b)
            icon.save(icon_path)
        
        # Set icon
        icon_img = tk.PhotoImage(file=icon_path)
        root.iconphoto(True, icon_img)
    except Exception:
        pass  # Ignore if can't set icon
    
    # Start the GUI loop
    root.mainloop()

if __name__ == "__main__":
    main() 