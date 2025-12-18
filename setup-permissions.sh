#!/bin/bash
# Setup script permissions for Linux
# Run this after cloning: chmod +x setup-permissions.sh && ./setup-permissions.sh

echo "Setting up script permissions..."

# Make all shell scripts executable
find . -name "*.sh" -type f -exec chmod +x {} \;

# Make Makefile executable (if needed)
chmod +x Makefile 2>/dev/null || true

echo "Done! All scripts are now executable."
echo ""
echo "Quick start:"
echo "  make install    # Install M.I.A"
echo "  make run        # Run M.I.A"
echo ""
echo "Or use the install script:"
echo "  ./scripts/install/install.sh"
