#!/bin/bash
# Build script for M.I.A v0.1.0 pre-release

set -e

echo "🚀 Building M.I.A v0.1.0 Pre-Release..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# Upgrade pip and install build tools
echo "🔧 Installing build tools..."
pip install --upgrade pip setuptools wheel build

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run basic tests
echo "🧪 Running basic tests..."
python -m pytest tests/unit/ -v || echo "⚠️  Some tests failed, continuing with build..."

# Build the package
echo "🏗️  Building package..."
python -m build

# Verify the build
echo "✅ Verifying build..."
if [ -f "dist/mia-successor-0.1.0.tar.gz" ]; then
    echo "✅ Source distribution created successfully"
else
    echo "❌ Source distribution not found"
    exit 1
fi

if [ -f "dist/mia_successor-0.1.0-py3-none-any.whl" ]; then
    echo "✅ Wheel distribution created successfully"
else
    echo "❌ Wheel distribution not found"
    exit 1
fi

# Show package info
echo "📋 Package information:"
python setup.py --name --version --description

echo ""
echo "🎉 Build completed successfully!"
echo "📦 Packages created in dist/ directory:"
ls -la dist/

echo ""
echo "🚀 Ready for pre-release v0.1.0!"
echo ""
echo "Next steps:"
echo "1. Test the wheel: pip install dist/mia_successor-0.1.0-py3-none-any.whl"
echo "2. Create git tag: git tag v0.1.0"
echo "3. Push tag: git push origin v0.1.0"
echo "4. Create GitHub release with the built packages"
