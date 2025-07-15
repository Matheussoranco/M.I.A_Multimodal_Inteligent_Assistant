# Requirements Guide for M.I.A

## Overview
This document explains the consolidated `requirements.txt` file for the M.I.A (Multimodal Intelligent Assistant) project.

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Requirements Categories

### Core ML and AI Dependencies
- **transformers**: Hugging Face transformers library
- **openai**: OpenAI API client
- **torch**: PyTorch machine learning framework
- **huggingface_hub**: Hugging Face model hub
- **langchain**: Language model application framework

### Audio Processing
- **sounddevice**: Real-time audio I/O
- **soundfile**: Audio file I/O
- **pydub**: Audio manipulation
- **pyttsx3**: Text-to-speech conversion
- **speechrecognition**: Speech recognition
- **PyAudio**: Audio I/O (commented out due to installation issues)

### Data Processing
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning library
- **datasets**: Dataset utilities
- **sentencepiece**: Text tokenization

### Web Automation
- **selenium**: Web browser automation
- **requests**: HTTP library

### System Utilities
- **psutil**: System and process monitoring
- **pyperclip**: Clipboard operations
- **plyer**: Platform-specific notifications

### Image Processing
- **Pillow**: Image processing library
- **matplotlib**: Plotting library

### Database and Memory
- **chromadb**: Vector database
- **networkx**: Network analysis

### Performance Monitoring
- **GPUtil**: GPU monitoring
- **memory-profiler**: Memory usage profiling

### Testing Framework
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel testing
- **pytest-mock**: Mocking utilities
- **pytest-benchmark**: Performance benchmarking

### Code Quality
- **black**: Code formatter
- **isort**: Import sorting
- **flake8**: Code linting
- **mypy**: Type checking
- **pylint**: Advanced linting
- **pydocstyle**: Docstring style checking

### Documentation
- **sphinx**: Documentation generation
- **sphinx-rtd-theme**: Read the Docs theme
- **docstring-parser**: Docstring parsing

### Development Tools
- **pre-commit**: Pre-commit hooks
- **tox**: Testing across environments
- **coverage**: Coverage measurement
- **jupyter**: Jupyter notebooks
- **ipython**: Enhanced Python shell
- **notebook**: Jupyter notebook server

### Debugging and Profiling
- **icecream**: Debugging output
- **line-profiler**: Line-by-line profiling
- **py-spy**: Sampling profiler

### Security Analysis
- **bandit**: Security analysis
- **safety**: Dependency security check

### Code Analysis
- **radon**: Code complexity analysis
- **prospector**: Code analysis toolkit
- **vulture**: Dead code detection

### Caching
- **redis**: Redis caching
- **diskcache**: Disk-based caching

### Testing Utilities
- **factory-boy**: Test data generation
- **faker**: Fake data generation
- **responses**: HTTP request mocking
- **freezegun**: Time mocking

### Platform-Specific Dependencies
- **win10toast**: Windows toast notifications (Windows only)
- **pywin32**: Windows API access (Windows only)

## Installation Tips

### Windows-Specific Issues
1. **PyAudio**: May require Visual C++ Build Tools
2. **torch**: Consider installing with CUDA support if you have a compatible GPU
3. **Some packages**: May require Microsoft Visual C++ 14.0 or greater

### macOS-Specific Issues
1. **sounddevice**: May require portaudio: `brew install portaudio`
2. **Some packages**: May require Xcode command line tools

### Linux-Specific Issues
1. **sounddevice**: May require ALSA development files
2. **PyQt5**: May require additional system packages

### Performance Optimization
1. **Optional dependencies**: Some packages are marked as optional for better performance
2. **GPU support**: Install CUDA-enabled versions of PyTorch if you have a compatible GPU
3. **Memory optimization**: Consider installing `numba` and `cython` for performance-critical code

## Dependency Management

### Updating Dependencies
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade package_name
```

### Checking for Security Issues
```bash
# Check for known security vulnerabilities
safety check
```

### Checking for Outdated Packages
```bash
# Check for outdated packages
pip list --outdated
```

## Troubleshooting

### Common Issues
1. **PyAudio installation fails**: Try installing system dependencies first
2. **CUDA not found**: Install CUDA toolkit or use CPU-only versions
3. **Permission denied**: Use `--user` flag or virtual environment
4. **Package conflicts**: Create fresh virtual environment

### Alternative Installation Methods
```bash
# Install with conda (if available)
conda install -c conda-forge package_name

# Install pre-compiled wheels
pip install --only-binary=all package_name
```

## Development Setup

### Full Development Environment
```bash
# Clone repository
git clone <repository_url>
cd M.I.A-The-successor-of-pseudoJarvis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run code quality checks
flake8 src/
black src/
mypy src/
```

### VS Code Integration
The requirements include all necessary packages for VS Code integration:
- Jupyter support
- Python debugging
- Code formatting
- Linting
- Testing

## Package Versions

All packages are pinned to minimum versions to ensure compatibility while allowing for updates. The format `package>=version` ensures you get at least that version but allows for newer compatible versions.

## License Considerations

Please review the licenses of all dependencies to ensure compliance with your project's licensing requirements. Some packages may have different license requirements for commercial use.

## Contributing

When adding new dependencies:
1. Add them to the appropriate section in `requirements.txt`
2. Update this guide with the new dependency
3. Test installation in a clean environment
4. Update the version if necessary

## Support

For issues related to specific packages, please refer to their respective documentation:
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Pytest Documentation](https://docs.pytest.org/)

For M.I.A-specific issues, please refer to the project's issue tracker.
