"""
User interface for the Cone Normal Map Generator.
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import math
import glob
import numpy as np
from numba import njit

from cone_normal_generator.config import (
    DEFAULT_SIZE, DEFAULT_DIAMETER, DEFAULT_HEIGHT, DEFAULT_STRENGTH,
    DEFAULT_SEGMENTS, DEFAULT_SEGMENT_RATIO, DEFAULT_MATCAP_ROTATION,
    AUTO_REFRESH_DELAY, SLIDER_DRAG_DELAY, TEMP_FOLDER, OUTPUT_FOLDER
)
from cone_normal_generator.core import ConeNormalMapGenerator
from cone_normal_generator.styling import DarkModeTheme, setup_dark_theme
from cone_normal_generator.helpers import validate_numeric, open_folder, clean_folder

# JIT-compiled function for stacked cone generation 
@njit
def _generate_stacked_cone_heightmap(size, num_cones, base_radius, spacing, height):
    """Generate stacked cone height map with Numba acceleration."""
    # Create an empty height map
    height_map = np.zeros((size, size), dtype=np.float32)
    
    # First cone position (center at -radius)
    first_x = -base_radius
    center_y = size - base_radius
    
    # Create stacked cones
    for i in range(num_cones):
        # Calculate position for this cone
        pos_x = first_x + (i * spacing)
        pos_y = center_y
        
        # Current parameters
        current_height = height
        current_radius = base_radius
        
        # Process each pixel
        for y in range(size):
            for x in range(size):
                # Calculate distance from center
                dist = np.sqrt((x - pos_x)**2 + (y - pos_y)**2)
                
                # Calculate cone height at this point
                if dist <= current_radius:
                    cone_val = (1.0 - dist / current_radius) * current_height
                    
                    # Only replace if this cone adds height
                    if cone_val > height_map[y, x]:
                        height_map[y, x] = cone_val
    
    return height_map

# JIT-compiled function for normal map calculation
@njit
def _stacked_heightmap_to_normal(height_map, strength, size):
    """Convert height map to normal map with JIT acceleration."""
    normal_map = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) * strength
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) * strength
    
    # Calculate gradients
    grad_x = np.zeros((size, size), dtype=np.float32)
    grad_y = np.zeros((size, size), dtype=np.float32)
    
    # Apply convolution manually
    for y in range(1, size-1):
        for x in range(1, size-1):
            # X gradient
            grad_x[y, x] = (
                sobel_x[0, 0] * height_map[y-1, x-1] +
                sobel_x[0, 1] * height_map[y-1, x] +
                sobel_x[0, 2] * height_map[y-1, x+1] +
                sobel_x[1, 0] * height_map[y, x-1] +
                sobel_x[1, 1] * height_map[y, x] +
                sobel_x[1, 2] * height_map[y, x+1] +
                sobel_x[2, 0] * height_map[y+1, x-1] +
                sobel_x[2, 1] * height_map[y+1, x] +
                sobel_x[2, 2] * height_map[y+1, x+1]
            )
            
            # Y gradient
            grad_y[y, x] = (
                sobel_y[0, 0] * height_map[y-1, x-1] +
                sobel_y[0, 1] * height_map[y-1, x] +
                sobel_y[0, 2] * height_map[y-1, x+1] +
                sobel_y[1, 0] * height_map[y, x-1] +
                sobel_y[1, 1] * height_map[y, x] +
                sobel_y[1, 2] * height_map[y, x+1] +
                sobel_y[2, 0] * height_map[y+1, x-1] +
                sobel_y[2, 1] * height_map[y+1, x] +
                sobel_y[2, 2] * height_map[y+1, x+1]
            )
    
    # Create the normal map
    for y in range(size):
        for x in range(size):
            # Map from [-1,1] to [0,255] range for RGB
            nx = 128 - grad_x[y, x] * 127
            ny = 128 - grad_y[y, x] * 127
            
            # Calculate Z component
            nz_squared = 1.0 - (grad_x[y, x]/127)**2 - (grad_y[y, x]/127)**2
            if nz_squared < 0:
                nz = 0
            else:
                nz = np.sqrt(nz_squared) * 255
                
            # Clamp values to valid range
            normal_map[y, x, 0] = min(max(int(nx), 0), 255)
            normal_map[y, x, 1] = min(max(int(ny), 0), 255)
            normal_map[y, x, 2] = min(max(int(nz), 0), 255)
    
    return normal_map

class ConeNormalMapApp:
    """Main application UI for the Cone Normal Map Generator."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Normal Perlage")  # Set the window title
        self.root.geometry("1400x800")
        self.root.configure(bg=DarkModeTheme.BACKGROUND)
        
        # Create core
        self.core = ConeNormalMapGenerator()
        
        # Configure style for dark mode
        self.style = setup_dark_theme(root)
        
        # Preview images placeholders
        self.height_preview = None
        self.normal_preview = None
        self.matcap_preview = None
        self.matcap_texture_preview = None
        self.empty_preview_img = None
        
        # UI state tracking
        self.slider_dragging = False
        self.stacked_slider_dragging = False  # Add state tracking for stacked tab
        self.update_timer = None
        self.stacked_update_timer = None  # Add timer for stacked tab
        self.use_full_resolution_stacked = False  # Track whether to use full resolution for stacked previews
        self.height_rotation_var = None  # Will be initialized in create_additional_tab_content
        
        # Matcap variables
        self.matcap_files = []
        self.matcap_var = tk.StringVar()
        self.stacked_matcap_var = tk.StringVar()
        
        # Flag for using Numba acceleration
        self.use_numba = True
        
        # Load available matcaps
        self.load_available_matcaps()
        
        # Create UI elements
        self.create_gui()
        
        # Force a full UI refresh before continguing
        self.root.update_idletasks()
        
        # Initialize UI with empty previews
        self.root.after(10, self.safe_display_empty_previews)
        
        # Start periodic updates
        self.root.after(100, self.start_periodic_updates)
        
        # When closing the app, clean up temp files
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_available_matcaps(self):
        """Load available matcap textures from assets folder."""
        # Define matcap directory
        matcap_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "matcaps")
        
        # Ensure the directory exists
        os.makedirs(matcap_dir, exist_ok=True)
        
        # Find all png files in the matcaps directory
        matcap_files = glob.glob(os.path.join(matcap_dir, "*.png"))
        
        # Extract just the filenames without paths for display
        self.matcap_files = [os.path.basename(f) for f in matcap_files]
        
        # Default to a basic matcap if none found
        if not self.matcap_files:
            print("No matcap files found in assets/matcaps directory")
            # The application will fall back to the default matcap in the core
        else:
            # Set initial value for both tabs
            self.matcap_var.set(self.matcap_files[0])
            self.stacked_matcap_var.set(self.matcap_files[0])
            print(f"Loaded {len(self.matcap_files)} matcap files from assets/matcaps directory")
    
    def on_matcap_selected(self, event=None):
        """Handle matcap selection change in the first tab."""
        selected_matcap = self.matcap_var.get()
        if not selected_matcap:
            return
            
        # Full path to the selected matcap
        matcap_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "assets", "matcaps", selected_matcap
        )
        
        try:
            # Load the matcap texture
            matcap_img = Image.open(matcap_path).convert("RGB")
            
            # Store in core for use in applying matcap effect
            self.core.matcap_texture = matcap_img
            
            # Update the matcap texture preview
            self.update_matcap_texture_preview()
            
            # If we already have a normal map, apply the new matcap
            if self.core.normal_image is not None:
                self.on_rotation_change()
                
            self.status_label.config(text=f"Status: Matcap changed to {selected_matcap}")
        except Exception as e:
            print(f"Error loading matcap: {e}")
            self.status_label.config(text=f"Status: Error loading matcap: {str(e)}")
    
    def on_stacked_matcap_selected(self, event=None):
        """Handle matcap selection change in the stacked tab."""
        selected_matcap = self.stacked_matcap_var.get()
        if not selected_matcap:
            return
            
        # Full path to the selected matcap
        matcap_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "assets", "matcaps", selected_matcap
        )
        
        try:
            # Load the matcap texture
            matcap_img = Image.open(matcap_path).convert("RGB")
            
            # Store in core for use in applying matcap effect
            self.core.matcap_texture = matcap_img
            
            # If we already have a normal map for stacked cones, apply the new matcap
            if hasattr(self, 'stacked_normal_image') and self.stacked_normal_image is not None:
                self.on_stacked_rotation_change()
                
            self.stacked_status_label.config(text=f"Matcap changed to {selected_matcap}")
        except Exception as e:
            print(f"Error loading matcap: {e}")
            self.stacked_status_label.config(text=f"Error loading matcap: {str(e)}")
    
    def create_gui(self):
        """Create the user interface elements."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.cone_tab = ttk.Frame(self.notebook)
        self.additional_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.cone_tab, text="Circular Engraving")
        self.notebook.add(self.additional_tab, text="Geneva Stripes")
        
        # Create content for cone tab
        self.create_cone_tab_content(self.cone_tab)
        
        # Create content for additional tab (placeholder for now)
        self.create_additional_tab_content(self.additional_tab)
        
        # Add a tab change handler
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def create_cone_tab_content(self, parent):
        """Create the content for the cone normal map tab."""
        # Left sidebar
        sidebar = ttk.Frame(parent, width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Make sidebar fixed width
        sidebar.pack_propagate(False)
        
        # Create parameter controls
        self.create_parameter_controls(sidebar)
        
        # Status section
        status_frame = ttk.Frame(sidebar)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready", wraplength=280)
        self.status_label.pack(pady=5, anchor=tk.W)
        
        # Right content for image display in a 2x2 grid
        content_frame = ttk.Frame(parent)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create preview canvases
        self.create_preview_canvases(content_frame)
    
    def create_additional_tab_content(self, parent):
        """Create the content for the additional normal map tab."""
        # Left sidebar
        sidebar = ttk.Frame(parent, width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Make sidebar fixed width
        sidebar.pack_propagate(False)
        
        # Parameters section
        parameters_frame = ttk.Frame(sidebar)
        parameters_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(parameters_frame, text="Stacked Cone Parameters", font=("Arial", 14)).pack(pady=(0, 10))
        ttk.Separator(parameters_frame).pack(fill=tk.X, pady=5)
        
        # Register validation commands
        validate_float = self.root.register(lambda P: validate_numeric(P, False))
        validate_int = self.root.register(lambda P: validate_numeric(P, True))
        
        # Size selection (reuse from first tab)
        size_frame = ttk.Frame(parameters_frame)
        size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(size_frame, text="Image Size:").pack(side=tk.LEFT, padx=5)
        
        size_values = ["128", "256", "512", "1024", "2048"]
        self.stacked_size_var = tk.StringVar(value=str(DEFAULT_SIZE))
        size_combo = ttk.Combobox(size_frame, textvariable=self.stacked_size_var, values=size_values, state="readonly", width=10)
        size_combo.pack(side=tk.LEFT, padx=5)
        size_combo.bind("<<ComboboxSelected>>", self.on_stacked_size_change)
        
        # Add some parameter sliders
        self.num_cones_var = tk.IntVar(value=5)
        
        # Create a wrapper function to ensure integer values for num_cones
        def on_num_cones_drag(val):
            # Force integer value during slider dragging
            int_val = int(float(val))
            if int_val != self.num_cones_var.get():
                self.num_cones_var.set(int_val)
            self.on_stacked_parameter_change()
        
        # Number of cones with custom integer handling
        num_cones_frame = ttk.Frame(parameters_frame)
        num_cones_frame.pack(fill=tk.X, pady=5)
        
        # Create a container for the label and auto-adjust info
        num_cones_header = ttk.Frame(num_cones_frame)
        num_cones_header.pack(fill=tk.X, padx=5)
        
        ttk.Label(num_cones_header, text="Number of Cones:").pack(side=tk.LEFT, padx=0)
        
        # This label will show the auto-adjusted value
        self.num_cones_auto_label = ttk.Label(num_cones_header, text="", foreground="#4296fa")
        self.num_cones_auto_label.pack(side=tk.RIGHT, padx=5)
        
        # Add a frame for the slider and input
        num_cones_control_frame = ttk.Frame(num_cones_frame)
        num_cones_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        num_cones_slider = ttk.Scale(num_cones_control_frame, from_=2, to=500, variable=self.num_cones_var, command=on_num_cones_drag)
        num_cones_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add number input
        num_cones_entry = ttk.Entry(num_cones_control_frame, textvariable=self.num_cones_var, width=5, validate="key", validatecommand=(validate_int, '%P'))
        num_cones_entry.pack(side=tk.RIGHT, padx=5)
        num_cones_entry.bind("<Return>", lambda e: self.on_stacked_parameter_change())
        num_cones_entry.bind("<FocusOut>", lambda e: self.on_stacked_parameter_change())
        
        # Add a "preferred spacing" entry that will help calculate optimal cone count
        spacing_frame = ttk.Frame(parameters_frame)
        spacing_frame.pack(fill=tk.X, pady=5)
        ttk.Label(spacing_frame, text="Preferred Spacing (%):").pack(anchor=tk.W, padx=5)
        
        # Add a frame for the slider and input
        spacing_control_frame = ttk.Frame(spacing_frame)
        spacing_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.preferred_spacing_var = tk.DoubleVar(value=5)  # Change to DoubleVar for decimal values
        spacing_slider = ttk.Scale(spacing_control_frame, from_=0.3, to=3, variable=self.preferred_spacing_var, command=lambda val: self.on_preferred_spacing_change())
        spacing_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add number input with float validation
        spacing_entry = ttk.Entry(spacing_control_frame, textvariable=self.preferred_spacing_var, width=5, validate="key", validatecommand=(validate_float, '%P'))
        spacing_entry.pack(side=tk.RIGHT, padx=5)
        spacing_entry.bind("<Return>", lambda e: self.on_preferred_spacing_change())
        spacing_entry.bind("<FocusOut>", lambda e: self.on_preferred_spacing_change())
        
        self.stacked_radius_var = tk.DoubleVar(value=100)
        self.create_parameter_slider(
            parameters_frame, "Cone Diameter (%):", self.stacked_radius_var,
            100, 500, validate_int, lambda e: self.on_stacked_parameter_change(), lambda: self.on_stacked_parameter_change()
        )
        
        self.stacked_height_var = tk.DoubleVar(value=DEFAULT_HEIGHT)
        self.create_parameter_slider(
            parameters_frame, "Cone Height:", self.stacked_height_var,
            0.1, 2.0, validate_float, lambda e: self.on_stacked_height_change(e), lambda: self.on_stacked_height_entry_change()
        )
        
        self.stacked_strength_var = tk.DoubleVar(value=DEFAULT_STRENGTH)
        self.create_parameter_slider(
            parameters_frame, "Normal Map Strength:", self.stacked_strength_var,
            5.0, 30.0, validate_float, lambda e: self.on_stacked_strength_change(e), lambda: self.on_stacked_strength_entry_change()
        )
        
        # Add height map rotation selector
        rotation_frame = ttk.Frame(parameters_frame)
        rotation_frame.pack(fill=tk.X, pady=5)
        ttk.Label(rotation_frame, text="Height Map Rotation:").pack(anchor=tk.W, padx=5)
        
        # Create a frame for the rotation options
        rotation_options_frame = ttk.Frame(rotation_frame)
        rotation_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create radio buttons for rotation options
        self.height_rotation_var = tk.IntVar(value=0)  # Default to 0 degrees
        
        # Create a row of radio buttons
        rotations = [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]
        for value, text in rotations:
            rb = ttk.Radiobutton(
                rotation_options_frame, 
                text=text, 
                variable=self.height_rotation_var, 
                value=value,
                command=self.on_height_rotation_change
            )
            rb.pack(side=tk.LEFT, padx=5, expand=True)
        
        # Add matcap rotation control
        self.stacked_rotation_var = tk.DoubleVar(value=DEFAULT_MATCAP_ROTATION)
        self.create_parameter_slider(
            parameters_frame, "Matcap Rotation (°):", self.stacked_rotation_var,
            0, 360, validate_float, lambda e: self.on_stacked_rotation_change(e), lambda: self.on_stacked_rotation_change()
        )
        
        # Add matcap selector
        matcap_selector_frame = ttk.Frame(parameters_frame)
        matcap_selector_frame.pack(fill=tk.X, pady=5)
        ttk.Label(matcap_selector_frame, text="Matcap Texture:").pack(anchor=tk.W, padx=5)
        
        # Create a combobox container
        matcap_combo_frame = ttk.Frame(matcap_selector_frame)
        matcap_combo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Show a dropdown if we have matcaps available
        matcap_combo = ttk.Combobox(
            matcap_combo_frame, 
            textvariable=self.stacked_matcap_var, 
            values=self.matcap_files,
            state="readonly" if self.matcap_files else "disabled",
            width=15
        )
        matcap_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        matcap_combo.bind("<<ComboboxSelected>>", self.on_stacked_matcap_selected)
        
        # Add fast preview checkbox
        fast_preview_frame = ttk.Frame(parameters_frame)
        fast_preview_frame.pack(fill=tk.X, pady=5)
        
        # Use the same variable as the main tab to stay in sync
        fast_preview_check = ttk.Checkbutton(
            fast_preview_frame, 
            text="Use Fast Preview", 
            variable=self.fast_preview_var,  # Reuse the same variable from the first tab
            command=self.on_stacked_fast_preview_change
        )
        fast_preview_check.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Add quality indicator label
        self.stacked_preview_quality_label = ttk.Label(
            fast_preview_frame, 
            text="(Fast Mode)" if self.fast_preview_var.get() else "(High Quality)", 
            foreground="#4296fa" if self.fast_preview_var.get() else "#f44336"
        )
        self.stacked_preview_quality_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Add grid line toggle for tiled preview
        grid_frame = ttk.Frame(parameters_frame)
        grid_frame.pack(fill=tk.X, pady=5)
        
        # Variable for gridline toggle
        self.show_grid_lines_var = tk.BooleanVar(value=False)
        grid_check = ttk.Checkbutton(
            grid_frame, 
            text="Show Grid Lines in Tiled Preview", 
            variable=self.show_grid_lines_var,
            command=self.on_grid_lines_toggle
        )
        grid_check.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create buttons
        buttons_frame = ttk.Frame(parameters_frame)
        buttons_frame.pack(fill=tk.X, pady=15)
        
        self.generate_stacked_btn = ttk.Button(
            buttons_frame, 
            text="Generate Stacked Cones", 
            command=self.on_stacked_generate, 
            style="Generate.TButton"
        )
        self.generate_stacked_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.save_stacked_btn = ttk.Button(
            buttons_frame, 
            text="Save Images", 
            command=self.on_stacked_save, 
            style="Save.TButton",
            state=tk.DISABLED
        )
        self.save_stacked_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Status section
        status_frame = ttk.Frame(sidebar)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.stacked_status_label = ttk.Label(status_frame, text="Status: Ready", wraplength=280)
        self.stacked_status_label.pack(pady=5, anchor=tk.W)
        
        # Right content for preview
        content_frame = ttk.Frame(parent)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create a simple preview area with a message
        preview_frame = ttk.Frame(content_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(
            preview_frame, 
            text="Stacked Cone Normal Map Mode", 
            font=("Arial", 18)
        ).pack(pady=(0, 10))
        
        # Create a grid for the previews
        preview_grid = ttk.Frame(preview_frame)
        preview_grid.pack(fill=tk.BOTH, expand=True)
        
        # Row 1: Height Map and Normal Map
        height_frame = ttk.Frame(preview_grid)
        height_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        normal_frame = ttk.Frame(preview_grid)
        normal_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Row 2: Matcap Preview and Pattern Visualization
        matcap_preview_frame = ttk.Frame(preview_grid)
        matcap_preview_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        tiled_preview_frame = ttk.Frame(preview_grid)
        tiled_preview_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure the grid to expand properly
        preview_grid.columnconfigure(0, weight=1)
        preview_grid.columnconfigure(1, weight=1)
        preview_grid.rowconfigure(0, weight=1)
        preview_grid.rowconfigure(1, weight=1)
        
        # Height map preview
        ttk.Label(height_frame, text="Stacked Height Map", font=("Arial", 12)).pack(pady=(0, 5))
        self.stacked_height_canvas = tk.Canvas(height_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.stacked_height_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Normal map preview
        ttk.Label(normal_frame, text="Stacked Normal Map", font=("Arial", 12)).pack(pady=(0, 5))
        self.stacked_normal_canvas = tk.Canvas(normal_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.stacked_normal_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Matcap preview
        ttk.Label(matcap_preview_frame, text="Stacked Matcap Preview", font=("Arial", 12)).pack(pady=(0, 5))
        self.stacked_matcap_preview_canvas = tk.Canvas(matcap_preview_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.stacked_matcap_preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tiled Matcap preview (4x4 grid)
        ttk.Label(tiled_preview_frame, text="Tiled Matcap Preview (4×4)", font=("Arial", 12)).pack(pady=(0, 5))
        self.tiled_matcap_canvas = tk.Canvas(tiled_preview_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.tiled_matcap_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize with placeholder text
        for canvas in [self.stacked_height_canvas, self.stacked_normal_canvas, 
                      self.stacked_matcap_preview_canvas, self.tiled_matcap_canvas]:
            canvas.create_text(
                200, 200, 
                text="Preview will be shown\nafter generation", 
                fill=DarkModeTheme.TEXT,
                font=("Arial", 14),
                justify=tk.CENTER
            )
        
        # Draw initial pattern visualization
        self.root.after(500, self.init_stacked_pattern)
    
    def init_stacked_pattern(self):
        """Initialize the stacked pattern with optimal values."""
        # Calculate optimal number of cones based on current parameters
        self.calculate_optimal_number_of_cones()
        
        # Initialize with an actual preview instead of just placeholder text
        self.root.after(100, lambda: self.generate_stacked_immediate_preview(use_full_resolution=False))
    
    def show_not_implemented(self):
        """Show a message that a feature is not implemented yet."""
        messagebox.showinfo("Not Implemented", "This feature is not yet implemented")
    
    def create_parameter_controls(self, parent):
        """Create all parameter control widgets."""
        # Parameters section
        parameters_frame = ttk.Frame(parent)
        parameters_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(parameters_frame, text="Cone Parameters", font=("Arial", 14)).pack(pady=(0, 10))
        ttk.Separator(parameters_frame).pack(fill=tk.X, pady=5)
        
        # Register validation commands
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
        
        # Create all the sliders with their input fields
        self.radius_var = tk.DoubleVar(value=DEFAULT_DIAMETER)
        self.create_parameter_slider(
            parameters_frame, "Cone Diameter (%):", self.radius_var,
            20, 180, validate_int, self.on_radius_change, self.on_radius_entry_change
        )
        
        self.height_var = tk.DoubleVar(value=DEFAULT_HEIGHT)
        self.create_parameter_slider(
            parameters_frame, "Cone Height:", self.height_var,
            0.1, 2.0, validate_float, self.on_height_change, self.on_height_entry_change
        )
        
        self.strength_var = tk.DoubleVar(value=DEFAULT_STRENGTH)
        self.create_parameter_slider(
            parameters_frame, "Normal Map Strength:", self.strength_var,
            1.0, 10.0, validate_float, self.on_strength_change, self.on_strength_entry_change
        )
        
        # Segments slider with custom behavior for integer values
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
        
        self.segment_ratio_var = tk.IntVar(value=DEFAULT_SEGMENT_RATIO)
        self.create_parameter_slider(
            parameters_frame, "Segment Ratio (% Out):", self.segment_ratio_var,
            10, 90, validate_int, self.on_segment_ratio_change, self.on_segment_ratio_entry_change
        )
        
        # Matcap section
        matcap_frame = ttk.Frame(parameters_frame)
        matcap_frame.pack(fill=tk.X, pady=10)
        ttk.Label(matcap_frame, text="Matcap Visualization", font=("Arial", 12)).pack(pady=(0, 5))
        ttk.Separator(matcap_frame).pack(fill=tk.X, pady=5)
        
        # Add matcap selector
        matcap_selector_frame = ttk.Frame(matcap_frame)
        matcap_selector_frame.pack(fill=tk.X, pady=5)
        ttk.Label(matcap_selector_frame, text="Matcap Texture:").pack(side=tk.LEFT, padx=5)
        
        # Show a dropdown if we have matcaps available
        matcap_combo = ttk.Combobox(
            matcap_selector_frame, 
            textvariable=self.matcap_var, 
            values=self.matcap_files,
            state="readonly" if self.matcap_files else "disabled",
            width=15
        )
        matcap_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        matcap_combo.bind("<<ComboboxSelected>>", self.on_matcap_selected)
        
        self.rotation_var = tk.DoubleVar(value=DEFAULT_MATCAP_ROTATION)
        self.create_parameter_slider(
            matcap_frame, "Matcap Rotation (°):", self.rotation_var,
            0, 360, validate_int, self.on_rotation_change, self.on_rotation_entry_change
        )
        
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
        self.create_action_buttons(parameters_frame)
    
    def create_parameter_slider(self, parent, label_text, variable, min_val, max_val, 
                              validate_cmd, slider_callback, entry_callback):
        """Create a slider with a label and input field."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        ttk.Label(frame, text=label_text).pack(anchor=tk.W, padx=5)
        
        # Add a frame for the slider and input
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        slider = ttk.Scale(control_frame, from_=min_val, to=max_val, variable=variable, command=slider_callback)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add number input
        entry = ttk.Entry(control_frame, textvariable=variable, width=5, validate="key", validatecommand=validate_cmd)
        entry.pack(side=tk.RIGHT, padx=5)
        entry.bind("<Return>", lambda e: entry_callback())
        entry.bind("<FocusOut>", lambda e: entry_callback())
        
        return frame, slider, entry
    
    def create_action_buttons(self, parent):
        """Create action buttons for the UI."""
        # Buttons
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill=tk.X, pady=15)
        
        self.generate_button = ttk.Button(buttons_frame, text="Generate", command=self.on_generate, style="Generate.TButton")
        self.generate_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.save_button = ttk.Button(buttons_frame, text="Save Images", command=self.on_save, style="Save.TButton", state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Clean output folder button
        clean_button_frame = ttk.Frame(parent)
        clean_button_frame.pack(fill=tk.X, pady=5)
        
        self.clean_button = ttk.Button(clean_button_frame, text="Clean Output Folder", command=self.on_clean_output, style="Clean.TButton")
        self.clean_button.pack(pady=5, fill=tk.X)
        
        # Output folder button
        folder_frame = ttk.Frame(parent)
        folder_frame.pack(fill=tk.X, pady=5)
        
        open_folder_button = ttk.Button(folder_frame, text="Open Output Folder", 
                                      command=lambda: self.open_output_folder())
        open_folder_button.pack(pady=5, fill=tk.X)
    
    def create_preview_canvases(self, parent):
        """Create canvas widgets for displaying the previews."""
        # Set up image preview area - 2x2 grid
        preview_frame = ttk.Frame(parent)
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
    
    def safe_display_empty_previews(self):
        """Thread-safe way to display empty previews on Tkinter canvases."""
        try:
            self.display_empty_previews()
            print("Empty previews initialized successfully")
        except Exception as e:
            print(f"Error initializing previews: {e}")
            # Try again after a short delay
            self.root.after(100, self.safe_display_empty_previews)
    
    def display_empty_previews(self):
        """Display empty preview canvases."""
        try:
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
        except Exception as e:
            print(f"Error displaying empty previews: {e}")
    
    # Parameter change handlers
    def on_size_change(self, event=None):
        """Handle change in size selection."""
        size = int(self.size_var.get())
        self.core.size = size
        
        # For resolution changes, we need a full regeneration, not just a fast preview
        # Clear any preview cache to force regeneration
        self.core.preview_size_cache = None
        self.core.preview_height_map = None
        
        # Use full generation instead of qguick preview for size changes
        if not self.core.generation_in_progress:
            self.core.generate_maps(self.update_gui)
    
    def on_radius_change(self, event=None):
        """Handle change in radius slider."""
        radius = round(self.radius_var.get())
        self.core.radius_percent = radius
        
        # Set dragging state while slider is being adjusted
        self.slider_dragging = True
        
        # Schedule preview update
        self.schedule_preview_update()
    
    def on_radius_entry_change(self):
        """Handle direct input in diameter entry."""
        try:
            diameter = int(self.radius_var.get())
            # Constrain to valid range
            diameter = max(20, min(180, diameter))
            self.radius_var.set(diameter)
            self.core.radius_percent = diameter
            
            # Not dragging when manually entering a value
            self.slider_dragging = False
            
            # Schedule preview update
            self.schedule_preview_update()
        except ValueError:
            # Reset to last valid value
            self.radius_var.set(self.core.radius_percent)
    
    def on_height_change(self, event=None):
        """Handle change in height slider."""
        height = round(self.height_var.get(), 1)
        self.core.height = height
        
        # Set dragging state
        self.slider_dragging = True
        
        # Schedule preview update
        self.schedule_preview_update()
    
    def on_height_entry_change(self):
        """Handle direct input in height entry."""
        try:
            height = float(self.height_var.get())
            # Constrain to valid range
            height = max(0.1, min(2.0, height))
            self.height_var.set(round(height, 1))
            self.core.height = height
            
            # Not dragging when manually entering a value
            self.slider_dragging = False
            
            # Schedule preview update
            self.schedule_preview_update()
        except ValueError:
            # Reset to last valid value
            self.height_var.set(self.core.height)
    
    def on_strength_change(self, event=None):
        """Handle change in strength slider."""
        strength = round(self.strength_var.get(), 1)
        self.core.strength = strength
        
        # Set dragging state
        self.slider_dragging = True
        
        # Schedule preview update
        self.schedule_preview_update()
    
    def on_strength_entry_change(self):
        """Handle direct input in strength entry."""
        try:
            strength = float(self.strength_var.get())
            # Constrain to valid range
            strength = max(1.0, min(10.0, strength))
            self.strength_var.set(round(strength, 1))
            self.core.strength = strength
            
            # Not dragging when manually entering a value
            self.slider_dragging = False
            
            # Schedule preview update
            self.schedule_preview_update()
        except ValueError:
            # Reset to last valid value
            self.strength_var.set(self.core.strength)
    
    def on_segments_change(self, event=None):
        """Handle change in segments slider."""
        # Ensure we're using an integer value
        segments = int(self.segments_var.get())
        self.segments_var.set(segments)  # Force integer in the UI
        self.core.segments = segments
        
        # Set dragging state
        self.slider_dragging = True
        
        # For segment changes, we need to clear the preview cache
        # to force a full regeneration of the height map
        self.core.preview_size_cache = None
        self.core.preview_height_map = None
        
        # Schedule preview update
        self.schedule_preview_update()
    
    def on_segments_entry_change(self):
        """Handle direct input in segments entry."""
        try:
            segments = int(self.segments_var.get())
            # Constrain to valid range
            segments = max(1, min(50, segments))
            self.segments_var.set(segments)
            self.core.segments = segments
            
            # Not dragging when manually entering a value
            self.slider_dragging = False
            
            # For segment changes, we need to clear the preview cache
            # to force a full regeneration of the height map
            self.core.preview_size_cache = None
            self.core.preview_height_map = None
            
            # Schedule preview update
            self.schedule_preview_update()
        except ValueError:
            # Reset to last valid value
            self.segments_var.set(self.core.segments)
    
    def on_segment_ratio_change(self, event=None):
        """Handle change in segment ratio slider."""
        ratio = int(self.segment_ratio_var.get())
        self.core.segment_ratio = ratio
        
        # Set dragging state
        self.slider_dragging = True
        
        # Schedule preview update
        self.schedule_preview_update()
    
    def on_segment_ratio_entry_change(self):
        """Handle direct input in segment ratio entry."""
        try:
            ratio = int(self.segment_ratio_var.get())
            # Constrain to valid range (10-90)
            ratio = max(10, min(90, ratio))
            self.segment_ratio_var.set(ratio)
            self.core.segment_ratio = ratio
            
            # Not dragging when manually entering a value
            self.slider_dragging = False
            
            # Schedule preview update
            self.schedule_preview_update()
        except ValueError:
            # Reset to last valid value
            self.segment_ratio_var.set(self.core.segment_ratio)
    
    def on_rotation_change(self, event=None):
        """Handle change in matcap rotation slider."""
        rotation = round(self.rotation_var.get())
        self.core.matcap_rotation = rotation
        
        # For rotation changes, we only need to update the matcap rendering,
        # not regenerate the normal map or height map.
        if (self.core.normal_image is not None and 
            self.core.matcap_texture is not None):
            
            # Use threading to avoid freezing the UI
            def update_rotation_thread():
                try:
                    # Apply matcap with current rotation
                    matcap_image = self.core.apply_matcap(
                        self.core.normal_image, 
                        rotation,
                        fast_preview=self.core.use_fast_preview
                    )
                    
                    if matcap_image:
                        # Update the matcap image in memory
                        self.core.matcap_image = matcap_image
                        
                        # Use after() to update UI from the main thread
                        self.root.after(10, self.update_matcap_preview)
                        
                except Exception as e:
                    print(f"Error updating matcap rotation: {str(e)}")
            
            # Start rotation update in background
            import threading
            thread = threading.Thread(target=update_rotation_thread)
            thread.daemon = True
            thread.start()
            
            # Set status to indicate processing
            self.status_label.config(text="Status: Updating matcap rotation preview...")
    
    def on_rotation_entry_change(self):
        """Handle direct input in rotation entry."""
        try:
            rotation = int(self.rotation_var.get())
            # Constrain to valid range (0-360)
            rotation = rotation % 360
            self.rotation_var.set(rotation)
            self.core.matcap_rotation = rotation
            
            # Update matcap preview
            if (self.core.normal_image is not None and 
                self.core.matcap_texture is not None):
                self.on_rotation_change()
        except ValueError:
            # Reset to last valid value
            self.rotation_var.set(self.core.matcap_rotation)
    
    def on_fast_preview_change(self, event=None):
        """Handle change in fast preview checkbox."""
        use_fast = self.fast_preview_var.get()
        self.core.use_fast_preview = use_fast
        
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
    
    def on_stacked_fast_preview_change(self, event=None):
        """Handle change in fast preview checkbox for stacked cones."""
        use_fast = self.fast_preview_var.get()
        self.core.use_fast_preview = use_fast
        
        # Update the quality indicator for the stacked tab
        if use_fast:
            self.stacked_preview_quality_label.config(text="(Fast Mode)", foreground="#4296fa")
        else:
            self.stacked_preview_quality_label.config(text="(High Quality)", foreground="#f44336")
        
        # Update status
        self.stacked_status_label.config(text=f"{'Fast preview mode' if use_fast else 'High quality mode'} selected.")
        
        # If we have a generated normal map, update the matcap preview
        if hasattr(self, 'stacked_normal_image') and self.stacked_normal_image is not None:
            self.on_stacked_rotation_change()
    
    # Button action handlers
    def on_generate(self):
        """Handle generate button click."""
        self.generate_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        
        # Start generation with update callback
        self.core.generate_maps(self.update_gui)
    
    def on_save(self):
        """Handle save button click."""
        if self.core.height_map is None or self.core.normal_map is None:
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
        height_file, normal_file = self.core.save_images(file_prefix)
        
        if height_file and normal_file:
            self.status_label.config(text=f"Status: Images saved successfully as {os.path.basename(height_file)} and {os.path.basename(normal_file)}")
    
    def on_clean_output(self):
        """Handle clean output folder button click."""
        # Ask for confirmation
        confirm = messagebox.askyesno(
            "Clean Output Folder", 
            f"Are you sure you want to delete all files in the {OUTPUT_FOLDER} folder?",
            icon=messagebox.WARNING
        )
        
        if confirm:
            result = clean_folder(OUTPUT_FOLDER)
            if result:
                self.status_label.config(text=f"Status: Cleaned output folder")
            else:
                self.status_label.config(text=f"Status: Error cleaning output folder")
    
    def open_output_folder(self):
        """Open the output folder in the file explorer."""
        result = open_folder(OUTPUT_FOLDER)
        if not result:
            self.status_label.config(text=f"Status: Error opening output folder")
    
    def on_closing(self):
        """Handle window closing event."""
        # Clean up temp folder if it exists
        clean_folder(TEMP_FOLDER)
        
        # Close the window
        self.root.destroy()
    
    # Preview update methods
    def update_gui(self):
        """Update UI with current core state."""
        # Use a try-except for every UI update to avoid thread issues
        try:
            # Update status
            self.status_label.config(text=f"Status: {self.core.status}")
            
            # Enable/disable buttons
            if self.core.generation_in_progress:
                self.generate_button.config(state=tk.DISABLED)
                self.save_button.config(state=tk.DISABLED)
            else:
                self.generate_button.config(state=tk.NORMAL)
                if self.core.height_map is not None and self.core.normal_map is not None:
                    self.save_button.config(state=tk.NORMAL)
            
            # Update each preview separately for better error handling
            if self.core.height_image is not None:
                self.update_height_preview()
            
            if self.core.normal_image is not None:
                self.update_normal_preview()
                
            if self.core.matcap_image is not None:
                self.update_matcap_preview()
                
            # Update matcap texture preview
            self.update_matcap_texture_preview()
        except Exception as e:
            print(f"Error in update_gui: {e}")
            # Try to refresh after a short delay
            self.root.after(100, self.update_gui)
    
    def update_height_preview(self):
        """Update height map preview."""
        if self.core.height_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.height_canvas.winfo_width()
        canvas_height = self.height_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
        
        # Calculate the display size to maintain aspect ratio
        display_size = min(canvas_width, canvas_height)
        
        # Resize image for display - use LANCZOS for better antialiasing in previews
        preview_img = self.core.height_image.resize((display_size, display_size), Image.LANCZOS)
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
        if self.core.normal_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.normal_canvas.winfo_width()
        canvas_height = self.normal_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
        
        # Calculate the display size to maintain aspect ratio
        display_size = min(canvas_width, canvas_height)
        
        # Resize image for display with LANCZOS antialiasing
        preview_img = self.core.normal_image.resize((display_size, display_size), Image.LANCZOS)
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
        if self.core.matcap_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.matcap_preview_canvas.winfo_width()
        canvas_height = self.matcap_preview_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
        
        # Calculate the display size to maintain aspect ratio
        display_size = min(canvas_width, canvas_height)
        
        # Resize image for display - use LANCZOS for better antialiasing in previews
        preview_img = self.core.matcap_image.resize((display_size, display_size), Image.LANCZOS)
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
        if self.core.matcap_texture is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.matcap_texture_canvas.winfo_width()
        canvas_height = self.matcap_texture_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
        
        # Calculate the display size to maintain aspect ratio
        display_size = min(canvas_width, canvas_height)
        
        # Get original matcap dimensions
        orig_width, orig_height = self.core.matcap_texture.size
        
        # Resize image for display - use LANCZOS for better antialiasing in previews
        # Handle different resolutions properly
        preview_img = self.core.matcap_texture.resize((display_size, display_size), Image.LANCZOS)
        photo = ImageTk.PhotoImage(preview_img)
        
        # Keep reference to avoid garbage collection
        self.matcap_texture_preview = photo
        
        # Calculate center position
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Clear canvas and display centered image
        self.matcap_texture_canvas.delete("all")
        self.matcap_texture_canvas.create_image(center_x, center_y, anchor=tk.CENTER, image=self.matcap_texture_preview)
        
        # Show resolution information
        self.matcap_texture_canvas.create_text(
            center_x, center_y + (display_size // 2) - 15,
            text=f"{orig_width}×{orig_height}",
            fill="#ffffff",
            font=("Arial", 9)
        )
    
    def start_periodic_updates(self):
        """Start the periodic UI update cycle."""
        self.root.after(100, self.periodic_update)
        
        # Generate an initial preview after UI is fully loaded
        if not self.core.generation_in_progress:
            self.root.after(500, lambda: self.core.qguick_update_preview(self.update_gui))
    
    def periodic_update(self):
        """Periodically update the UI."""
        # Update UI with current core state
        self.update_gui()
        
        # Reschedule
        self.root.after(100, self.periodic_update)
    
    def schedule_preview_update(self):
        """Schedule a preview update with a delay to avoid too frequent updates."""
        # Cancel any existing timer
        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        
        # Use longer delay during slider dragging for better performance
        delay = SLIDER_DRAG_DELAY if self.slider_dragging else AUTO_REFRESH_DELAY
        
        # Create a new timer
        self.update_timer = self.root.after(delay, self.on_delayed_update)
    
    def on_delayed_update(self):
        """Called after delay to update preview."""
        if not self.core.generation_in_progress:
            # Pass the slider_dragging flag to use appropriate preview scaling
            self.core.qguick_update_preview(
                update_callback=self.update_gui,
                is_dragging=self.slider_dragging
            )
            
        # Reset the dragging flag after update
        self.slider_dragging = False
    
    def create_stacked_preview_canvases(self, parent):
        """Create canvas widgets for displaying the stacked cone previews."""
        # Set up image preview area - 2x2 grid
        preview_frame = ttk.Frame(parent)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a grid for the previews
        preview_grid = ttk.Frame(preview_frame)
        preview_grid.pack(fill=tk.BOTH, expand=True)
        
        # Row 1: Height Map and Normal Map
        height_frame = ttk.Frame(preview_grid)
        height_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        normal_frame = ttk.Frame(preview_grid)
        normal_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Row 2: Matcap Preview and Pattern Visualization
        matcap_preview_frame = ttk.Frame(preview_grid)
        matcap_preview_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        tiled_preview_frame = ttk.Frame(preview_grid)
        tiled_preview_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure the grid to expand properly
        preview_grid.columnconfigure(0, weight=1)
        preview_grid.columnconfigure(1, weight=1)
        preview_grid.rowconfigure(0, weight=1)
        preview_grid.rowconfigure(1, weight=1)
        
        # Height map preview
        ttk.Label(height_frame, text="Stacked Height Map", font=("Arial", 12)).pack(pady=(0, 5))
        self.stacked_height_canvas = tk.Canvas(height_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.stacked_height_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Normal map preview
        ttk.Label(normal_frame, text="Stacked Normal Map", font=("Arial", 12)).pack(pady=(0, 5))
        self.stacked_normal_canvas = tk.Canvas(normal_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.stacked_normal_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Matcap preview
        ttk.Label(matcap_preview_frame, text="Stacked Matcap Preview", font=("Arial", 12)).pack(pady=(0, 5))
        self.stacked_matcap_preview_canvas = tk.Canvas(matcap_preview_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.stacked_matcap_preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tiled Matcap preview (4x4 grid)
        ttk.Label(tiled_preview_frame, text="Tiled Matcap Preview (4×4)", font=("Arial", 12)).pack(pady=(0, 5))
        self.tiled_matcap_canvas = tk.Canvas(tiled_preview_frame, bg=DarkModeTheme.DARK_BG, highlightthickness=1, highlightbackground=DarkModeTheme.BORDER)
        self.tiled_matcap_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize with placeholder text
        for canvas in [self.stacked_height_canvas, self.stacked_normal_canvas, 
                      self.stacked_matcap_preview_canvas, self.tiled_matcap_canvas]:
            canvas.create_text(
                200, 200, 
                text="Preview will be shown\nafter generation", 
                fill=DarkModeTheme.TEXT,
                font=("Arial", 14),
                justify=tk.CENTER
            )
        
        # Draw initial pattern visualization
        self.root.after(500, self.init_stacked_pattern)
    
    def on_stacked_parameter_change(self):
        """Handle changes to any of the stacked cone parameters."""
        # Force integer value for num_cones
        num_cones = int(self.num_cones_var.get())
        self.num_cones_var.set(num_cones)  # Force integer in the UI
        
        # If diameter or size changed, recalculate optimal cone count
        if hasattr(self, '_last_radius') and self._last_radius != self.stacked_radius_var.get():
            self.calculate_optimal_number_of_cones()
        elif hasattr(self, '_last_size') and self._last_size != self.stacked_size_var.get():
            self.calculate_optimal_number_of_cones()
        
        # Store current values for comparison on next change
        self._last_radius = self.stacked_radius_var.get()
        self._last_size = self.stacked_size_var.get()
        
        # Set dragging state
        self.stacked_slider_dragging = True
        
        # Schedule preview update, keeping the current resolution mode
        self.schedule_stacked_preview_update()
    
    def on_stacked_generate(self):
        """Handle generate button click for stacked cones."""
        # Update status
        self.stacked_status_label.config(text="Generating stacked cone normal map...")
        
        # Get parameters
        size = int(self.stacked_size_var.get())
        height = self.stacked_height_var.get()
        strength = self.stacked_strength_var.get()
        
        # Store current size as the last used value
        self._last_size = size
        
        try:
            # Calculate the optimal number of cones and diameter
            # This updates self.stacked_radius_var and self.num_cones_var
            optimal_num_cones, actual_spacing = self.calculate_optimal_number_of_cones()
            
            # Use the adjusted radius percentage directly (already updated in the variable)
            adjusted_radius_percent = self.stacked_radius_var.get()
            
            # Generate the actual height map and normal map
            self.generate_stacked_cone_maps(size, optimal_num_cones, adjusted_radius_percent, height, strength)
            
            # Enable the save button after successful generation
            self.save_stacked_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.stacked_status_label.config(text=f"Error generating maps: {str(e)}")
            print(f"Error in stacked cone generation: {e}")
    
    def on_stacked_save(self):
        """Handle save button click for stacked cones."""
        if not hasattr(self, 'stacked_height_image') or not hasattr(self, 'stacked_normal_image'):
            messagebox.showinfo("No Images", "Please generate stacked cone images first")
            return
            
        # Ask for file prefix
        initial_dir = os.path.abspath(OUTPUT_FOLDER)
        file_prefix = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            title="Save Stacked Cone Normal Map Images",
            filetypes=[("PNG files", "*.png")],
            defaultextension=".png"
        )
        
        if not file_prefix:
            return
            
        # Remove extension if user added one
        file_prefix = os.path.splitext(file_prefix)[0]
        
        # Save the images
        height_file = f"{file_prefix}_stacked_height_map.png"
        normal_file = f"{file_prefix}_stacked_normal_map.png"
        
        try:
            self.stacked_height_image.save(height_file)
            self.stacked_normal_image.save(normal_file)
            self.stacked_status_label.config(
                text=f"Saved as {os.path.basename(height_file)} and {os.path.basename(normal_file)}"
            )
        except Exception as e:
            self.stacked_status_label.config(text=f"Error saving files: {str(e)}")
            print(f"Error saving stacked cone images: {e}")
            
    def generate_stacked_cone_maps(self, size, num_cones, radius_percent, height, strength, is_preview=False):
        """Generate stacked cone height and normal maps with repeatable edges."""
        # Start timing
        import time
        start_time = time.time()
        
        # Calculate base radius from the optimized radius_percent
        # We don't need to round again since it was already optimized
        base_radius = size * radius_percent / 100 / 2
        base_diameter = base_radius * 2
        
        # Calculate total pattern width
        total_pattern_width = size + 2 * base_radius
        
        # Calculate spacing based on the number of cones and total pattern width
        if num_cones > 1:
            spacing = total_pattern_width / (num_cones - 1)
            spacing_percent = round((spacing / total_pattern_width) * 100, 2)
        else:
            spacing = 0
            spacing_percent = 0
        
        # Use Numba acceleration if enabled
        if self.use_numba:
            # Use JIT-accelerated function for height map generation
            height_map = _generate_stacked_cone_heightmap(size, num_cones, base_radius, spacing, height)
        else:
            # Original non-JIT code
            # Create an empty height map
            height_map = np.zeros((size, size), dtype=np.float32)
            
            # First cone position (center at -radius)
            first_x = -base_radius
            center_y = size - base_radius
            
            # Create stacked cones
            for i in range(num_cones):
                # Calculate position for this cone
                pos_x = first_x + (i * spacing)
                pos_y = center_y
                
                # All cones have the same height and radius
                current_height = height
                current_radius = base_radius
                
                # Generate the cone (even if partially off-image)
                y_indices, x_indices = np.ogrid[:size, :size]
                dist_from_center = np.sqrt((x_indices - pos_x)**2 + (y_indices - pos_y)**2)
                
                # Calculate cone height at each point (only within image bounds)
                cone_height = np.maximum(0, 1.0 - dist_from_center / current_radius) * current_height
                
                # Combine with existing height map using 'replace' blend mode where the cone exists
                mask = cone_height > 0  # Create a mask where the cone exists
                height_map[mask] = cone_height[mask]  # Replace only where the cone exists
        
        # Convert to PIL image for rotation
        height_img = Image.fromarray((height_map / np.max(height_map) * 255).astype(np.uint8))
        
        # Apply rotation to height map if needed
        rotation_angle = self.height_rotation_var.get()
        if rotation_angle != 0:
            # PIL's rotate method takes counterclockwise angles, so we negate the angle
            height_img = height_img.rotate(-rotation_angle, resample=Image.BICUBIC, expand=False)
            # Convert back to numpy array for normal map generation
            height_map = np.array(height_img).astype(np.float32) / 255.0 * np.max(height_map)
        
        # Scale strength based on resolution
        resolution_factor = size / 512.0  # Normalize to a reference resolution of 512
        adjusted_strength = strength * resolution_factor
        
        if self.use_numba:
            # Use JIT-accelerated normal map calculation
            normal_map = _stacked_heightmap_to_normal(height_map, adjusted_strength, size)
            normal_img = Image.fromarray(normal_map)
        else:
            # Calculate gradients using Sobel operators (more accurate at higher resolutions)
            normal_map = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Apply Sobel kernels for better gradient calculation
            from scipy import ndimage
            
            # Create Sobel kernels
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            # Apply convolution for gradient calculation
            grad_x = ndimage.convolve(height_map, sobel_x)
            grad_y = ndimage.convolve(height_map, sobel_y)
            
            # Scale gradients by strength
            grad_x = grad_x * adjusted_strength
            grad_y = grad_y * adjusted_strength
            
            # Create the normal map
            normal_map[:, :, 0] = np.uint8(np.clip(128 - grad_x * 127, 0, 255))
            normal_map[:, :, 1] = np.uint8(np.clip(128 - grad_y * 127, 0, 255))
            
            # Calculate Z component ensuring it's normalized
            z_component = np.sqrt(np.maximum(0.0, 1.0 - np.square(grad_x/127) - np.square(grad_y/127)))
            normal_map[:, :, 2] = np.uint8(np.clip(z_component * 255, 0, 255))
            
            # Convert to PIL image
            normal_img = Image.fromarray(normal_map)
        
        # Generate matcap preview if matcap texture is available
        matcap_img = None
        if hasattr(self.core, 'matcap_texture') and self.core.matcap_texture is not None:
            # Use the core's apply_matcap function with current rotation setting
            matcap_img = self.core.apply_matcap(
                normal_img, 
                self.stacked_rotation_var.get(),
                fast_preview=self.core.use_fast_preview
            )
        
        # Report timing for profiling
        end_time = time.time()
        if not is_preview:
            print(f"Stacked cone map generation took {end_time - start_time:.3f} seconds")
            if self.use_numba:
                print("Numba acceleration was used")
            else:
                print("Numba acceleration was NOT used")
        
        # Save current images in memory
        self.stacked_height_image = height_img
        self.stacked_normal_image = normal_img
        self.stacked_matcap_image = matcap_img
        
        return height_img, normal_img, matcap_img
    
    def update_stacked_previews(self):
        """Update the stacked cone preview canvases with the generated images."""
        # Only update if images exist
        if not hasattr(self, 'stacked_height_image') or not hasattr(self, 'stacked_normal_image'):
            return
            
        # Get canvas dimensions for height map
        canvas_width = self.stacked_height_canvas.winfo_width()
        canvas_height = self.stacked_height_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
        
        # Calculate display size
        display_size = min(canvas_width, canvas_height)
        
        # Update height map preview
        # Use LANCZOS for better antialiasing in previews
        height_img = self.stacked_height_image.resize((display_size, display_size), Image.LANCZOS)
        height_photo = ImageTk.PhotoImage(height_img)
        self.stacked_height_photo = height_photo  # Keep reference to avoid garbage collection
        
        # Calculate center position and display
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        self.stacked_height_canvas.delete("all")
        self.stacked_height_canvas.create_image(center_x, center_y, image=height_photo, anchor=tk.CENTER)
        
        # Update normal map preview with LANCZOS antialiasing
        normal_img = self.stacked_normal_image.resize((display_size, display_size), Image.LANCZOS)
        normal_photo = ImageTk.PhotoImage(normal_img)
        self.stacked_normal_photo = normal_photo  # Keep reference
        
        self.stacked_normal_canvas.delete("all")
        self.stacked_normal_canvas.create_image(center_x, center_y, image=normal_photo, anchor=tk.CENTER)
        
        # Update matcap preview with LANCZOS antialiasing
        if hasattr(self, 'stacked_matcap_image') and self.stacked_matcap_image is not None:
            matcap_img = self.stacked_matcap_image.resize((display_size, display_size), Image.LANCZOS)
            matcap_photo = ImageTk.PhotoImage(matcap_img)
            self.stacked_matcap_photo = matcap_photo  # Keep reference
            
            self.stacked_matcap_preview_canvas.delete("all")
            self.stacked_matcap_preview_canvas.create_image(center_x, center_y, image=matcap_photo, anchor=tk.CENTER)
            
            # Update the 4x4 tiled matcap preview
            self.update_tiled_matcap_preview()
    
    def on_grid_lines_toggle(self):
        """Handle toggle of grid lines in tiled preview."""
        # If we have a matcap image, update the tiled preview
        if hasattr(self, 'stacked_matcap_image'):
            self.update_tiled_matcap_preview()
            
    def update_tiled_matcap_preview(self):
        """Create and display a 4x4 tiled matcap preview showing seamless tiling."""
        if not hasattr(self, 'stacked_matcap_image'):
            return
            
        # Get canvas dimensions
        canvas_width = self.tiled_matcap_canvas.winfo_width()
        canvas_height = self.tiled_matcap_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
            
        # Calculate the size of each tile in the 4x4 grid
        tile_size = min(canvas_width, canvas_height) // 4
        
        # Create a new image for the 4x4 grid
        grid_size = tile_size * 4
        tiled_img = Image.new('RGB', (grid_size, grid_size), color=(0, 0, 0))
        
        # Resize the matcap image to the tile size
        # For tiling, we still need to use NEAREST for the edges to match perfectly
        # But for preview purposes, we can use LANCZOS for the inner area of each tile
        
        # Create a slightly larger tile with LANCZOS for high quality
        larger_size = tile_size + 2
        large_tile = self.stacked_matcap_image.resize((larger_size, larger_size), Image.LANCZOS)
        
        # Then crop it to the exact tile size to maintain seamless edges
        matcap_tile = large_tile.crop((1, 1, 1 + tile_size, 1 + tile_size))
        
        # Paste the matcap image in a 4x4 grid
        for y in range(4):
            for x in range(4):
                tiled_img.paste(matcap_tile, (x * tile_size, y * tile_size))
        
        # Only add grid lines if the toggle is on
        if self.show_grid_lines_var.get():
            # Add grid lines to show the tile boundaries
            draw = ImageDraw.Draw(tiled_img)
            
            # Draw vertical and horizontal lines at tile boundaries
            for i in range(1, 4):
                # Vertical lines
                draw.line([(i * tile_size, 0), (i * tile_size, grid_size)], fill=(60, 60, 60), width=1)
                # Horizontal lines
                draw.line([(0, i * tile_size), (grid_size, i * tile_size)], fill=(60, 60, 60), width=1)
        
        # Convert to PhotoImage and display
        tiled_photo = ImageTk.PhotoImage(tiled_img)
        self.tiled_matcap_photo = tiled_photo  # Keep reference
        
        # Calculate center position
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Clear canvas and display centered image
        self.tiled_matcap_canvas.delete("all")
        self.tiled_matcap_canvas.create_image(center_x, center_y, image=tiled_photo, anchor=tk.CENTER)
        
        # Add explanatory text
        if self.show_grid_lines_var.get():
            title_text = "4×4 Grid Showing Seamless Tiling (With Grid Lines)"
        else:
            title_text = "4×4 Grid Showing Seamless Tiling (No Grid Lines)"
            
        self.tiled_matcap_canvas.create_text(
            center_x, 15,
            text=title_text,
            fill="#ff6b6b",
            font=("Arial", 9)
        )
    
    def on_stacked_rotation_change(self, event=None):
        """Handle change in matcap rotation for stacked cones."""
        # Only update if we already have a normal map generated
        if not hasattr(self, 'stacked_normal_image'):
            return
            
        # Get the current rotation value
        rotation = round(self.stacked_rotation_var.get())
        
        # Generate a new matcap preview with the updated rotation
        if hasattr(self.core, 'matcap_texture') and self.core.matcap_texture is not None:
            # Use threading to avoid freezing the UI
            def update_stacked_rotation_thread():
                try:
                    # Apply matcap with current rotation
                    matcap_image = self.core.apply_matcap(
                        self.stacked_normal_image, 
                        rotation,
                        fast_preview=self.core.use_fast_preview
                    )
                    
                    if matcap_image:
                        # Update the matcap image in memory
                        self.stacked_matcap_image = matcap_image
                        
                        # Use after() to update UI from the main thread
                        self.root.after(10, self.update_stacked_matcap_preview)
                        
                except Exception as e:
                    print(f"Error updating stacked matcap rotation: {str(e)}")
            
            # Start rotation update in background
            import threading
            thread = threading.Thread(target=update_stacked_rotation_thread)
            thread.daemon = True
            thread.start()
            
            # Set status to indicate processing
            self.stacked_status_label.config(text="Updating matcap rotation preview...")
        else:
            # If we don't have a normal map yet, schedule a preview update
            # which will generate everything from scratch
            self.stacked_slider_dragging = True
            
            # Schedule preview update, keeping the current resolution mode
            self.schedule_stacked_preview_update()
    
    def update_stacked_matcap_preview(self):
        """Update only the matcap preview for stacked cones."""
        if not hasattr(self, 'stacked_matcap_image'):
            return
            
        # Get canvas dimensions
        canvas_width = self.stacked_matcap_preview_canvas.winfo_width()
        canvas_height = self.stacked_matcap_preview_canvas.winfo_height()
        
        if canvas_width < 50:  # Not properly sized yet
            canvas_width = 300
            canvas_height = 300
        
        # Calculate display size
        display_size = min(canvas_width, canvas_height)
        
        # Resize image for display with LANCZOS antialiasing
        matcap_img = self.stacked_matcap_image.resize((display_size, display_size), Image.LANCZOS)
        matcap_photo = ImageTk.PhotoImage(matcap_img)
        self.stacked_matcap_photo = matcap_photo  # Keep reference
        
        # Calculate center position
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Clear canvas and display centered image
        self.stacked_matcap_preview_canvas.delete("all")
        self.stacked_matcap_preview_canvas.create_image(center_x, center_y, image=matcap_photo, anchor=tk.CENTER)
        
        # Also update the tiled preview
        self.update_tiled_matcap_preview()
        
        # Update status
        self.stacked_status_label.config(text="Matcap rotation updated.")
    
    def on_preferred_spacing_change(self):
        """Handle changes to the preferred spacing slider."""
        # Calculate optimal number of cones based on preferred spacing
        self.calculate_optimal_number_of_cones()
        
        # Set dragging state
        self.stacked_slider_dragging = True
        
        # Schedule preview update, keeping the current resolution mode
        self.schedule_stacked_preview_update()
    
    def calculate_optimal_number_of_cones(self):
        """Calculate the optimal number of cones based on diameter and preferred spacing."""
        # Get current parameters
        size = int(self.stacked_size_var.get())
        radius_percent = self.stacked_radius_var.get()
        preferred_spacing_percent = self.preferred_spacing_var.get()
        
        # Calculate base radius & diameter in pixels (raw, unrounded values)
        base_radius_raw = size * radius_percent / 100 / 2
        
        # Define a range of acceptable radius values to consider around the user's selection
        # Try radius values within ±10% of the user's selection, but at least ±1 pixel
        radius_range = max(1, base_radius_raw * 0.1)
        
        # For very large radius values, constrain the range to be reasonable
        if radius_range > 10:
            radius_range = 10
            
        min_radius = max(1, base_radius_raw - radius_range)
        max_radius = base_radius_raw + radius_range
        
        # Maximum number of cones to consider
        max_reasonable_cones = 500
        
        # Minimum spacing in pixels
        min_spacing_pixels = 1
        
        # To track the best combination
        best_solutions = []
        
        # For each possible radius value (consider only integer radius values)
        # Since we need to check many values, we'll only check integer radii for efficiency
        for test_radius in range(round(min_radius), round(max_radius) + 1):
            # Skip invalid radius values
            if test_radius <= 0:
                continue
                
            # Calculate total pattern width with this radius
            total_pattern_width = size + 2 * test_radius
            
            # Convert preferred spacing to pixels
            preferred_spacing_pixels = (preferred_spacing_percent / 100) * total_pattern_width
            
            # Ensure minimum spacing
            if preferred_spacing_pixels < min_spacing_pixels:
                preferred_spacing_pixels = min_spacing_pixels
            
            # For each possible number of cones
            for test_cones in range(2, max_reasonable_cones + 1):
                # Calculate spacing for this number of cones
                test_spacing = total_pattern_width / (test_cones - 1)
                
                # Only consider solutions where the spacing is close to preferred
                spacing_ratio = test_spacing / preferred_spacing_pixels
                if 0.8 <= spacing_ratio <= 1.2:
                    # For seamless tiling, spacing must be integer or very close
                    if abs(test_spacing - round(test_spacing)) < 0.01:
                        # Calculate how far this radius is from the user's preferred radius
                        radius_error = abs(test_radius - base_radius_raw)
                        # Calculate how far this spacing is from the preferred spacing
                        spacing_error = abs(test_spacing - preferred_spacing_pixels)
                        
                        # Calculate a combined error score (weighted)
                        # Give radius error 2x weight since it affects the look more
                        combined_error = (2 * radius_error / base_radius_raw) + (spacing_error / preferred_spacing_pixels)
                        
                        # Add to possible solutions
                        best_solutions.append({
                            'radius': test_radius,
                            'radius_percent': (test_radius * 2 / size) * 100,
                            'num_cones': test_cones,
                            'spacing': test_spacing,
                            'spacing_percent': (test_spacing / total_pattern_width) * 100,
                            'error': combined_error
                        })
        
        # Sort solutions by the combined error
        best_solutions.sort(key=lambda x: x['error'])
        
        # Use the best solution if available
        if best_solutions:
            best_match = best_solutions[0]
            optimal_num_cones = best_match['num_cones']
            optimal_radius = best_match['radius']
            optimal_radius_percent = best_match['radius_percent']
            actual_spacing_percent = best_match['spacing_percent']
            
            # Update the UI with the new diameter value
            self.stacked_radius_var.set(round(optimal_radius_percent, 2))
        else:
            # Fallback if no good solutions found - just round to nearest
            optimal_radius = round(base_radius_raw)
            optimal_radius_percent = (optimal_radius * 2 / size) * 100
            
            # Use the original logic for fallback cone count
            total_pattern_width = size + 2 * optimal_radius
            preferred_spacing_pixels = (preferred_spacing_percent / 100) * total_pattern_width
            exact_num_cones = ((total_pattern_width / preferred_spacing_pixels) + 1)
            optimal_num_cones = max(2, min(500, round(exact_num_cones)))
            
            actual_spacing_pixels = total_pattern_width / (optimal_num_cones - 1)
            actual_spacing_percent = (actual_spacing_pixels / total_pattern_width) * 100
            
            # Update the UI with the rounded diameter
            self.stacked_radius_var.set(round(optimal_radius_percent, 2))
        
        # Update the UI with the selected number of cones
        self.num_cones_var.set(optimal_num_cones)
        
        # Show exact spacing result with appropriate decimal places
        formatted_spacing = "{:.2f}".format(round(actual_spacing_percent, 2))
        formatted_diameter = "{:.2f}".format(round(optimal_radius_percent, 2))
        spacing_text = f"(Auto: {optimal_num_cones} @ {formatted_spacing}%)"
        self.num_cones_auto_label.config(text=spacing_text)
        
        # Update status with explanation of the adjustment
        status_text = f"Adjusted to {optimal_num_cones} cones with {formatted_diameter}% diameter for perfect seamless tiling."
        self.stacked_status_label.config(text=status_text)
        
        return optimal_num_cones, round(actual_spacing_percent, 2) 
    
    def schedule_stacked_preview_update(self):
        """Schedule a preview update for the stacked cones with a delay to avoid too frequent updates."""
        # Cancel any existing timer
        if hasattr(self, 'stacked_update_timer') and self.stacked_update_timer is not None:
            self.root.after_cancel(self.stacked_update_timer)
        
        # Use longer delay during slider dragging for better performance
        delay = SLIDER_DRAG_DELAY if hasattr(self, 'stacked_slider_dragging') and self.stacked_slider_dragging else AUTO_REFRESH_DELAY
        
        # Create a new timer
        self.stacked_update_timer = self.root.after(delay, self.on_stacked_delayed_update)
    
    def on_stacked_delayed_update(self):
        """Called after delay to update stacked cones preview."""
        try:
            # Get current parameters
            size = int(self.stacked_size_var.get())
            height = self.stacked_height_var.get()
            strength = self.stacked_strength_var.get()
            
            # Calculate optimal number of cones and diameter
            optimal_num_cones, actual_spacing = self.calculate_optimal_number_of_cones()
            
            # Use the adjusted radius percentage directly
            adjusted_radius_percent = self.stacked_radius_var.get()
            
            # Always respect the user's full resolution preference, regardless of dragging state
            if self.use_full_resolution_stacked:
                # Use full resolution
                preview_size = size
                is_preview = False
                self.stacked_status_label.config(text=f"Generating full resolution {size}×{size} px preview...")
            else:
                # Use a smaller preview size for optimization
                preview_size = min(256, size)
                if size > 512:
                    preview_size = 256  # Keep preview small for large output sizes
                is_preview = True
                # Update the status to show we're doing a quick preview
                self.stacked_status_label.config(text=f"Generating quick preview at {preview_size}×{preview_size} px...")
            
            # Generate a preview with the appropriate parameters
            self.generate_stacked_cone_maps(
                preview_size, 
                optimal_num_cones, 
                adjusted_radius_percent, 
                height, 
                strength,
                is_preview=is_preview
            )
            
            # Update the enabled state of the save button based on the preview mode
            if not is_preview:
                self.save_stacked_btn.config(state=tk.NORMAL)
                self.stacked_status_label.config(
                    text=f"Generated at {size}×{size} px"
                )
            else:
                # Only disable save button for small previews
                if preview_size < size:
                    self.save_stacked_btn.config(state=tk.DISABLED)
                    self.stacked_status_label.config(
                        text=f"Preview at {preview_size}×{preview_size} px (Output size: {size}×{size} px)"
                    )
                else:
                    self.save_stacked_btn.config(state=tk.NORMAL)
                    self.stacked_status_label.config(
                        text=f"Generated at {size}×{size} px"
                    )
            
            # Reset dragging state
            self.stacked_slider_dragging = False
                
        except Exception as e:
            print(f"Error in stacked preview generation: {e}")
            self.stacked_status_label.config(text=f"Preview error: {str(e)}")
            self.stacked_slider_dragging = False
    
    def on_stacked_size_change(self, event=None):
        """Handle change in size selection for stacked cones."""
        size = int(self.stacked_size_var.get())
        
        # Store current value for comparison
        self._last_size = size
        
        # For size changes, recalculate optimal cone count
        self.calculate_optimal_number_of_cones()
        
        # Set dragging state to false since this is an explicit selection
        self.stacked_slider_dragging = False
        
        # For resolution changes, we need a more immediate update
        # Don't just schedule a delayed update, generate immediately
        self.stacked_status_label.config(text=f"Updating to {size}×{size} resolution...")
        
        # If we have an active timer, cancel it
        if hasattr(self, 'stacked_update_timer') and self.stacked_update_timer is not None:
            self.root.after_cancel(self.stacked_update_timer)
        
        # When changing resolution, we always want to use full resolution
        self.use_full_resolution_stacked = True
        print(f"Size changed to {size}px, set full resolution mode to True")
        
        # Generate preview immediately with full resolution (not a preview size)
        self.root.after(10, lambda: self.generate_stacked_immediate_preview(use_full_resolution=True))
    
    def on_stacked_height_change(self, event=None):
        """Handle change in height slider for stacked cones."""
        height = round(self.stacked_height_var.get(), 1)
        
        # Set dragging state
        self.stacked_slider_dragging = True
        
        # Print the current resolution mode (debugging)
        print(f"Height change, use_full_resolution_stacked: {self.use_full_resolution_stacked}")
        
        # Schedule preview update, keeping the current resolution mode
        self.schedule_stacked_preview_update()
    
    def on_stacked_height_entry_change(self):
        """Handle direct input in height entry for stacked cones."""
        try:
            height = float(self.stacked_height_var.get())
            # Constrain to valid range
            height = max(0.1, min(2.0, height))
            self.stacked_height_var.set(round(height, 1))
            
            # Not dragging when manually entering a value
            self.stacked_slider_dragging = False
            
            # Schedule preview update
            self.schedule_stacked_preview_update()
        except ValueError:
            # Reset to last valid value
            self.stacked_height_var.set(DEFAULT_HEIGHT)
    
    def on_stacked_strength_change(self, event=None):
        """Handle change in strength slider for stacked cones."""
        strength = round(self.stacked_strength_var.get(), 1)
        
        # Set dragging state
        self.stacked_slider_dragging = True
        
        # Print the current resolution mode (debugging)
        print(f"Strength change, use_full_resolution_stacked: {self.use_full_resolution_stacked}")
        
        # Schedule preview update, keeping the current resolution mode
        self.schedule_stacked_preview_update()
    
    def on_stacked_strength_entry_change(self):
        """Handle direct input in strength entry for stacked cones."""
        try:
            strength = float(self.stacked_strength_var.get())
            # Constrain to valid range
            strength = max(5.0, min(30.0, strength))
            self.stacked_strength_var.set(round(strength, 1))
            
            # Not dragging when manually entering a value
            self.stacked_slider_dragging = False
            
            # Schedule preview update
            self.schedule_stacked_preview_update()
        except ValueError:
            # Reset to last valid value
            self.stacked_strength_var.set(DEFAULT_STRENGTH)
    
    def generate_stacked_immediate_preview(self, use_full_resolution=False):
        """Generate an immediate preview for the stacked cones after size change.
        
        Args:
            use_full_resolution: If True, use the actual selected size instead of a preview size
        """
        try:
            # Update the resolution mode tracking
            self.use_full_resolution_stacked = use_full_resolution
            print(f"generate_stacked_immediate_preview setting use_full_resolution_stacked = {use_full_resolution}")
            
            # Get current parameters
            size = int(self.stacked_size_var.get())
            height = self.stacked_height_var.get()
            strength = self.stacked_strength_var.get()
            
            # Get the optimal values
            optimal_num_cones, actual_spacing = self.calculate_optimal_number_of_cones()
            adjusted_radius_percent = self.stacked_radius_var.get()
            
            # For size changes, use the actual size if requested, otherwise use a smaller preview size
            if use_full_resolution:
                preview_size = size  # Use full resolution
                self.stacked_status_label.config(text=f"Generating full resolution {size}×{size} px preview...")
            else:
                # Use a smaller preview size for performance
                preview_size = min(256, size)
                if size > 512:
                    preview_size = 256  # Keep preview small for large output sizes
                self.stacked_status_label.config(text=f"Generating quick preview at {preview_size}×{preview_size} px...")
            
            # Generate a preview with the current parameters
            self.generate_stacked_cone_maps(
                preview_size, 
                optimal_num_cones, 
                adjusted_radius_percent, 
                height, 
                strength,
                is_preview=not use_full_resolution  # If using full resolution, it's not a "preview"
            )
            
            # Update status to reflect preview or full resolution
            if use_full_resolution:
                self.stacked_status_label.config(
                    text=f"Generated at {size}×{size} px"
                )
                # Enable save button for full resolution renders
                self.save_stacked_btn.config(state=tk.NORMAL)
            else:
                # Only show preview message if the size is different from full resolution
                if preview_size < size:
                    self.stacked_status_label.config(
                        text=f"Preview at {preview_size}×{preview_size} px (Output size: {size}×{size} px)"
                    )
                    # Disable save button for small previews
                    self.save_stacked_btn.config(state=tk.DISABLED)
                else:
                    # Small resolutions like 128x128 will be full size even in preview mode
                    self.stacked_status_label.config(
                        text=f"Generated at {size}×{size} px"
                    )
                    self.save_stacked_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            print(f"Error generating immediate preview: {e}")
            self.stacked_status_label.config(text=f"Preview error: {str(e)}")
    
    def on_tab_changed(self, event=None):
        """Handle when the user switches between tabs."""
        current_tab = self.notebook.index(self.notebook.select())
        
        # If switching to the stacked cones tab (index 1), refresh preview
        if current_tab == 1:
            # Reset any pending previews
            if hasattr(self, 'stacked_update_timer') and self.stacked_update_timer is not None:
                self.root.after_cancel(self.stacked_update_timer)
                
            # Generate a fresh preview if we don't already have one
            if not hasattr(self, 'stacked_normal_image') or self.stacked_normal_image is None:
                # Always start with low-res preview when first switching to tab
                self.use_full_resolution_stacked = False
                self.root.after(100, lambda: self.generate_stacked_immediate_preview(use_full_resolution=False))
    
    def on_height_rotation_change(self):
        """Handle changes to the height map rotation."""
        # Get the selected rotation angle
        rotation = self.height_rotation_var.get()
        
        # Update status
        self.stacked_status_label.config(text=f"Height map rotation set to {rotation}°")
        
        # If we already have a height map, regenerate the preview with the new rotation
        if hasattr(self, 'stacked_height_image') and self.stacked_height_image is not None:
            # Schedule a preview update with current resolution mode
            self.schedule_stacked_preview_update()