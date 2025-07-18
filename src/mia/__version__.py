"""
Version information for M.I.A - Multimodal Intelligent Assistant
"""

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

# Release information
__title__ = "M.I.A"
__description__ = "Multimodal Intelligent Assistant"
__author__ = "Matheus Pullig Soranço de Carvalho"
__author_email__ = "matheussoranco@gmail.com"
__license__ = "AGPLv3"
__url__ = "https://github.com/Matheussoranco/M.I.A-The-successor-of-pseudoJarvis"

# Build information
__build__ = "pre-release"
__status__ = "Development"

def get_version():
    """Get the version string."""
    return __version__

def get_version_info():
    """Get the version info tuple."""
    return __version_info__

def get_full_version():
    """Get full version information."""
    return {
        "version": __version__,
        "title": __title__,
        "description": __description__,
        "author": __author__,
        "license": __license__,
        "build": __build__,
        "status": __status__
    }
