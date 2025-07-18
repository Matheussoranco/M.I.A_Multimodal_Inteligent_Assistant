from setuptools import setup, find_packages
import os
import sys

# Add src to path to import version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from mia.__version__ import __version__, __author__, __author_email__, __description__, __url__, __license__

def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove version constraints that might cause issues
                    if '>=' in line:
                        requirements.append(line)
                    elif '==' in line:
                        # Convert exact versions to minimum versions for flexibility
                        package = line.split('==')[0]
                        version = line.split('==')[1]
                        requirements.append(f"{package}>={version}")
                    else:
                        requirements.append(line)
            return requirements
    return []

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mia-successor",
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=__author__,
    author_email=__author_email__,
    url=__url__,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'mia = mia.main:main',
        ],
    },
    include_package_data=True,
    license=__license__,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
)
