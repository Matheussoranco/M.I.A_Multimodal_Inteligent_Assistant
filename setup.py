from setuptools import setup, find_packages

setup(
    name="mia-successor",
    version="0.1.0",
    description="Friday/M.I.A - Your Personal Virtual Assistant powered by LLMs and modular automation.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/friday-mia",
    packages=find_packages(),
    install_requires=[
        # Dependencies are read from requirements.txt
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'mia = main_modules.main:main',
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
