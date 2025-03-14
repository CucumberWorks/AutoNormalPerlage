"""
Core generator functionality for cone normal maps.
"""
import os
import time
import threading
import math
import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter
from numba import njit, float32, float64, int32, boolean

from cone_normal_generator.config import (
    DEFAULT_SIZE, DEFAULT_HEIGHT, DEFAULT_STRENGTH, DEFAULT_DIAMETER,
    DEFAULT_MATCAP_ROTATION, DEFAULT_SEGMENTS, DEFAULT_SEGMENT_RATIO,
    FAST_PREVIEW_SCALE, DRAG_PREVIEW_SCALE,
    TEMP_FOLDER, OUTPUT_FOLDER, ASSETS_FOLDER
)
from cone_normal_generator.helpers import ensure_folders_exist

# Numba-optimized functions
@njit
def _create_standard_cone(size, center_x, center_y, radius, height, dist_from_center):
    """Create a standard cone height map using Numba acceleration."""
    # Create the standard cone shape (height decreases linearly with distance)
    height_map = np.maximum(0, height * (1.0 - dist_from_center / radius))
    return height_map

@njit
def _create_segmented_cone(size, radius, height, segments, segment_ratio, dist_from_center):
    """Create a segmented cone height map using Numba acceleration."""
    # Normalize distance to be 0 to 1 within the radius
    normalized_dist = np.clip(dist_from_center / radius, 0, 1)
    
    # Calculate position within the segment cycle (0 to 1 for each complete segment)
    segment_pos = (normalized_dist * segments) % 1.0
    
    # Create height map
    height_map = np.zeros_like(normalized_dist)
    
    # Process each pixel
    for y in range(size):
        for x in range(size):
            pos = segment_pos[y, x]
            # Check if in up-slope (out) portion
            if pos < segment_ratio:
                # For up-slope: height increases linearly from 0 to 1
                height_map[y, x] = pos / segment_ratio
            else:
                # For down-slope: height decreases linearly from 1 to 0
                height_map[y, x] = 1.0 - (pos - segment_ratio) / (1.0 - segment_ratio)
    
    # Fade out to 0 at the edges and scale by height
    height_map = height * height_map * (1.0 - normalized_dist)
    
    # Set values outside the radius to 0
    height_map[normalized_dist >= 1.0] = 0
    
    return height_map

@njit
def _apply_sobel_kernels(height_map_float, kernel_x, kernel_y):
    """Apply Sobel kernels using Numba acceleration."""
    height, width = height_map_float.shape
    dx = np.zeros_like(height_map_float)
    dy = np.zeros_like(height_map_float)
    
    # Apply the kernels manually
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Apply x kernel
            dx[y, x] = (
                -1 * height_map_float[y-1, x-1] + 1 * height_map_float[y-1, x+1] +
                -2 * height_map_float[y, x-1] + 2 * height_map_float[y, x+1] +
                -1 * height_map_float[y+1, x-1] + 1 * height_map_float[y+1, x+1]
            ) * kernel_x[1, 1]  # Use center value as scale factor
            
            # Apply y kernel
            dy[y, x] = (
                -1 * height_map_float[y-1, x-1] + -2 * height_map_float[y-1, x] + -1 * height_map_float[y-1, x+1] +
                1 * height_map_float[y+1, x-1] + 2 * height_map_float[y+1, x] + 1 * height_map_float[y+1, x+1]
            ) * kernel_y[1, 1]  # Use center value as scale factor
            
    return dx, dy

class ConeNormalMapGenerator:
    """Core generator class for creating cone normal maps."""
    
    def __init__(self):
        self.size = DEFAULT_SIZE
        self.height = DEFAULT_HEIGHT
        self.strength = DEFAULT_STRENGTH
        self.radius_percent = DEFAULT_DIAMETER  # Keeping variable name for compatibility, but using diameter value
        self.matcap_rotation = DEFAULT_MATCAP_ROTATION
        self.segments = DEFAULT_SEGMENTS
        self.segment_ratio = DEFAULT_SEGMENT_RATIO
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
            normal_map_small = normal_map.resize((preview_size, preview_size), Image.BILINEAR)
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
            result_img = result_img.resize((normal_map.width, normal_map.height), Image.BILINEAR)
            return result_img
        else:
            return Image.fromarray(matcap_result)

    def create_cone_height_map(self, fast_preview=False, drag_preview=False):
        """Create a height map of a cone shape with optional segmentation."""
        # If in fast preview mode, use a smaller size
        actual_size = self.size
        if fast_preview and self.use_fast_preview:
            # Use an even smaller size during dragging for better performance
            preview_scale = DRAG_PREVIEW_SCALE if drag_preview else FAST_PREVIEW_SCALE
            actual_size = max(int(self.size * preview_scale), 64)  # Minimum size of 64px
            
            # Check if we can reuse the cached preview
            if self.preview_size_cache == (actual_size, self.radius_percent, self.height, self.segments, self.segment_ratio):
                return self.preview_height_map
            
        size = actual_size
        center = (size // 2, size // 2)
        
        # Convert diameter percentage to radius percentage (divide by 2)
        radius_percent = self.radius_percent / 2.0
        radius = int(size * (radius_percent / 100.0))
        
        height = self.height
        segments = self.segments
        segment_ratio = self.segment_ratio / 100.0  # Convert from percentage (0-100) to fraction (0-1)
        
        # Create a grid of coordinates
        y, x = np.ogrid[:size, :size]
        
        # Calculate distance from center for each pixel
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        if segments <= 1:
            # Use Numba-accelerated function for standard cone
            height_map = _create_standard_cone(size, center[0], center[1], radius, height, dist_from_center)
        else:
            # Use Numba-accelerated function for segmented cone
            height_map = _create_segmented_cone(size, radius, height, segments, segment_ratio, dist_from_center)
        
        # Cache for fast preview mode
        if fast_preview and self.use_fast_preview:
            self.preview_size_cache = (actual_size, self.radius_percent, self.height, self.segments, self.segment_ratio)
            self.preview_height_map = height_map
        
        return height_map

    def height_map_to_normal_map(self, height_map):
        """Convert a height map to a normal map using SciPy with Numba acceleration."""
        # Get the dimensions of the height map
        height, width = height_map.shape
        
        # Scale strength based on resolution for consistent results between different sizes
        # Using DEFAULT_SIZE as the reference size - we need to increase strength for higher resolutions
        resolution_scale = max(width, height) / DEFAULT_SIZE
        adjusted_strength = self.strength * resolution_scale
        
        # Define Sobel kernels
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * adjusted_strength
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) * adjusted_strength
        
        # Convert height_map to float32 for better precision
        height_map_float = height_map.astype(np.float32)
        
        # Use Numba-accelerated Sobel application for gradient calculation
        dx, dy = _apply_sobel_kernels(height_map_float, kernel_x, kernel_y)
        
        # Prepare normal vector components
        z = np.ones_like(dx)
        
        # Stack the components for vectorized operations
        normal_vectors = np.stack([dx, dy, z], axis=-1)
        
        # Normalize to get unit vectors (vectorized)
        # Add small epsilon to avoid division by zero
        normal_length = np.sqrt(np.sum(normal_vectors**2, axis=2, keepdims=True)) + 1e-10
        normal_vectors_normalized = normal_vectors / normal_length
        
        # Map from [-1,1] to [0,255] range for RGB
        normal_map = np.uint8((normal_vectors_normalized + 1.0) * 127.5)
        
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
        if (self.height_map is None and self.height_image is None) or (self.normal_map is None and self.normal_image is None):
            print("No maps to save - both height and normal maps are required")
            return None, None
        
        try:
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
            
            # Always regenerate at full resolution for saving
            self.status = "Generating full resolution height map for saving..."
            
            # Generate full resolution height map
            full_res_height_map = self.create_cone_height_map(fast_preview=False)
            
            # Generate full resolution normal map
            self.status = "Generating full resolution normal map for saving..."
            full_res_normal_map = self.height_map_to_normal_map(full_res_height_map)
            
            # Create height map image
            height_vis = (full_res_height_map / np.max(full_res_height_map) * 255).astype(np.uint8)
            height_img = Image.fromarray(height_vis, mode='L')
            height_img.save(height_file)
            
            # Create normal map image
            normal_img = Image.fromarray(full_res_normal_map, mode='RGB')
            normal_img.save(normal_file)
            
            # Save status message
            status_msg = f"Images saved at full resolution: {height_file} and {normal_file}"
            
            # If we have a matcap texture, also save the matcap visualization
            if self.matcap_texture is not None:
                # Always generate high-quality matcap render at full resolution
                matcap_img = self.apply_matcap(normal_img, self.matcap_rotation, fast_preview=False)
                if matcap_img:
                    matcap_img.save(matcap_file)
                    status_msg += f" and {matcap_file}"
            
            # Update the instance variables with the full resolution data
            self.height_map = full_res_height_map
            self.normal_map = full_res_normal_map
            
            # Update the preview images with the full resolution data
            self.height_image = height_img
            self.normal_image = normal_img
            
            # Save current version to output folder too
            self.height_image.save(self.current_height_file)
            self.normal_image.save(self.current_normal_file)
            
            # Update matcap preview with full resolution
            if self.matcap_texture is not None and matcap_img:
                self.matcap_image = matcap_img
                self.matcap_image.save(self.current_matcap_file)
            
            self.status = status_msg
            
            return height_file, normal_file
        except Exception as e:
            self.status = f"Error saving: {str(e)}"
            print(f"Error saving images: {str(e)}")
            return None, None

    def qguick_update_preview(self, update_callback=None, is_dragging=False):
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
                fast_height_map = self.create_cone_height_map(fast_preview=use_fast, drag_preview=is_dragging)
                
                # Update the height image from the height map
                if fast_height_map is not None:
                    # Store the height map for saving as well
                    self.height_map = fast_height_map
                    
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
                    # Store the normal map for saving as well
                    self.normal_map = fast_normal_map
                    
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