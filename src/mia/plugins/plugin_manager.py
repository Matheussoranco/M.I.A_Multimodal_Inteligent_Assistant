"""
Plugin Manager: Dynamically loads and manages plugins/tools for new skills.
"""
import importlib
import os
from pathlib import Path

class PluginManager:
    def __init__(self, plugin_dir=None):
        if plugin_dir is None:
            # Default to the plugins directory relative to this file
            plugin_dir = Path(__file__).parent
        self.plugin_dir = plugin_dir
        self.plugins = {}

    def load_plugins(self):
        """Load all Python plugins from the plugins directory."""
        try:
            plugin_files = []
            if os.path.exists(self.plugin_dir):
                for fname in os.listdir(self.plugin_dir):
                    if fname.endswith(".py") and fname != "__init__.py" and fname != "plugin_manager.py":
                        plugin_files.append(fname)
            
            for fname in plugin_files:
                try:
                    name = fname[:-3]
                    # Use relative import since we're in the plugins package
                    module = importlib.import_module(f".{name}", package="src.mia.plugins")
                    self.plugins[name] = module
                except ImportError as e:
                    print(f"Warning: Could not load plugin {name}: {e}")
                    
        except Exception as e:
            print(f"Warning: Could not load plugins: {e}")
            # Continue without plugins

    def get_plugin(self, name):
        return self.plugins.get(name)
