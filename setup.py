from setuptools import setup, find_packages
import os

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
    version="0.1.0",
    description="M.I.A - Your Personal Virtual Assistant powered by LLMs and modular automation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matheus Pullig Soranço de Carvalho",
    author_email="matheussoranco@gmail.com",
    url="https://github.com/Matheussoranco/M.I.A-The-successor-of-pseudoJarvis",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'mia = src.mia.main:main',
        ],
    },
    include_package_data=True,
    license="AGPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
)
