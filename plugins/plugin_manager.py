"""
Plugin Manager: Dynamically loads and manages plugins/tools for new skills.
"""
import importlib
import os

class PluginManager:
    def __init__(self, plugin_dir="plugins"):
        self.plugin_dir = plugin_dir
        self.plugins = {}

    def load_plugins(self):
        for fname in os.listdir(self.plugin_dir):
            if fname.endswith(".py") and fname != "__init__.py":
                name = fname[:-3]
                module = importlib.import_module(f"plugins.{name}")
                self.plugins[name] = module

    def get_plugin(self, name):
        return self.plugins.get(name)
