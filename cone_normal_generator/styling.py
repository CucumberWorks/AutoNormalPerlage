"""
UI styling and theme for the Cone Normal Map Generator.
"""
import tkinter as tk
from tkinter import ttk

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
    TAB_BACKGROUND = "#282828"
    TAB_ACTIVE = "#3a3a3a"
    TAB_SELECTED = "#464646"

def setup_dark_theme(root):
    """Apply dark mode theme to the application.
    
    Args:
        root: The Tkinter root window
    """
    style = ttk.Style()
    
    # Configure styling
    style.theme_use('default')  # Start with default
    
    # Configure colors
    style.configure('TFrame', background=DarkModeTheme.BACKGROUND)
    style.configure('TLabel', background=DarkModeTheme.BACKGROUND, foreground=DarkModeTheme.TEXT)
    style.configure('TButton', background=DarkModeTheme.BUTTON_BG, foreground=DarkModeTheme.TEXT)
    style.map('TButton', 
              background=[('active', DarkModeTheme.BUTTON_ACTIVE)],
              foreground=[('active', DarkModeTheme.TEXT)])
    
    # Scale style
    style.configure('TScale', background=DarkModeTheme.BACKGROUND, 
                   troughcolor=DarkModeTheme.SLIDER_BG)
    
    # Combobox style
    style.configure('TCombobox', 
                   fieldbackground=DarkModeTheme.DARK_BG,
                   background=DarkModeTheme.BUTTON_BG,
                   foreground=DarkModeTheme.TEXT,
                   arrowcolor=DarkModeTheme.TEXT)
    
    # Map additional states for combobox
    style.map('TCombobox',
             fieldbackground=[('readonly', DarkModeTheme.DARK_BG)],
             background=[('readonly', DarkModeTheme.BUTTON_BG)],
             foreground=[('readonly', DarkModeTheme.TEXT)])
    
    # Notebook (tabs) style
    style.configure('TNotebook', background=DarkModeTheme.BACKGROUND, 
                  borderwidth=0)
    style.configure('TNotebook.Tab', background=DarkModeTheme.TAB_BACKGROUND,
                  foreground=DarkModeTheme.TEXT,
                  padding=[10, 5],
                  borderwidth=0)
    
    style.map('TNotebook.Tab',
            background=[('selected', DarkModeTheme.TAB_SELECTED), 
                         ('active', DarkModeTheme.TAB_ACTIVE)],
            foreground=[('selected', DarkModeTheme.TEXT)])
    
    # Fix for combobox dropdown text color in dark mode
    root.option_add('*TCombobox*Listbox.background', DarkModeTheme.DARK_BG)
    root.option_add('*TCombobox*Listbox.foreground', DarkModeTheme.TEXT)
    root.option_add('*TCombobox*Listbox.selectBackground', DarkModeTheme.ACCENT)
    root.option_add('*TCombobox*Listbox.selectForeground', DarkModeTheme.TEXT)
    
    # Special button styles
    style.configure('Generate.TButton', 
                   background=DarkModeTheme.GREEN_BUTTON, 
                   foreground=DarkModeTheme.TEXT)
    style.map('Generate.TButton', 
             background=[('active', DarkModeTheme.GREEN_BUTTON_ACTIVE)])
    
    style.configure('Save.TButton', 
                   background=DarkModeTheme.BLUE_BUTTON, 
                   foreground=DarkModeTheme.TEXT)
    style.map('Save.TButton', 
             background=[('active', DarkModeTheme.BLUE_BUTTON_ACTIVE)])
    
    # Clean button style
    style.configure('Clean.TButton', 
                   background="#964B00", 
                   foreground=DarkModeTheme.TEXT)
    style.map('Clean.TButton', 
             background=[('active', "#B25900")])
    
    return style 