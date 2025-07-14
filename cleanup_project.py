#!/usr/bin/env python3
"""
Cleanup Script for M.I.A Project
Removes old, redundant, and useless files after reorganization
"""

import os
import shutil
import sys
from pathlib import Path

def cleanup_old_files():
    """Remove old files and directories that are no longer needed"""
    base_dir = Path(".")
    
    # Old directories to remove (now moved to src/mia/)
    old_dirs = [
        "audio",
        "core", 
        "deployment",
        "langchain",
        "learning",
        "llm",
        "main_modules",
        "memory",
        "multimodal",
        "planning",
        "plugins",
        "security",
        "system",
        "tools",
        "utils"
    ]
    
    # Old files to remove (now moved to scripts/ or docs/)
    old_files = [
        "run.bat",
        "run-audio.bat", 
        "run-mixed.bat",
        "run-text-only.bat",
        "run.sh",
        "install.bat",
        "install.sh",
        "install_ffmpeg.bat",
        "uninstall.sh",
        "check-system.sh",
        "dev.sh",
        "quickstart.sh",
        "test_ollama.py",
        "README.md",
        "CHANGELOG.md",
        "USAGE.md",
        "USAGE_GUIDE.md",
        ".env.example"
    ]
    
    # Temporary files to remove
    temp_files = [
        "reorganize_project.py",
        "create_init_files.py",
        "REORGANIZATION_COMPLETE.md"
    ]
    
    # Build directories to remove
    build_dirs = [
        "mia_successor.egg-info",
        "__pycache__",
        "*.pyc",
        ".pytest_cache"
    ]
    
    print("M.I.A Project Cleanup")
    print("=" * 40)
    print("\\nRemoving old directories...")
    
    for dir_name in old_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                print(f"✓ Removed directory: {dir_name}/")
            except Exception as e:
                print(f"✗ Error removing {dir_name}/: {e}")
        else:
            print(f"- Directory not found: {dir_name}/")
    
    print("\\nRemoving old files...")
    
    for file_name in old_files:
        file_path = base_dir / file_name
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                print(f"✓ Removed file: {file_name}")
            except Exception as e:
                print(f"✗ Error removing {file_name}: {e}")
        else:
            print(f"- File not found: {file_name}")
    
    print("\\nRemoving temporary files...")
    
    for file_name in temp_files:
        file_path = base_dir / file_name
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                print(f"✓ Removed temp file: {file_name}")
            except Exception as e:
                print(f"✗ Error removing {file_name}: {e}")
        else:
            print(f"- Temp file not found: {file_name}")
    
    print("\\nRemoving build artifacts...")
    
    for pattern in build_dirs:
        if pattern == "mia_successor.egg-info":
            dir_path = base_dir / pattern
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    print(f"✓ Removed build directory: {pattern}")
                except Exception as e:
                    print(f"✗ Error removing {pattern}: {e}")
    
    # Clean up __pycache__ directories recursively
    for root, dirs, files in os.walk(base_dir):
        if '__pycache__' in dirs:
            pycache_path = Path(root) / '__pycache__'
            try:
                shutil.rmtree(pycache_path)
                print(f"✓ Removed __pycache__: {pycache_path}")
            except Exception as e:
                print(f"✗ Error removing __pycache__: {e}")
    
    # Remove .pyc files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pyc'):
                pyc_path = Path(root) / file
                try:
                    pyc_path.unlink()
                    print(f"✓ Removed .pyc file: {pyc_path}")
                except Exception as e:
                    print(f"✗ Error removing .pyc file: {e}")

def check_before_cleanup():
    """Check what files will be removed before cleanup"""
    base_dir = Path(".")
    
    print("Files and directories that will be removed:")
    print("=" * 50)
    
    # Check old directories
    old_dirs = [
        "audio", "core", "deployment", "langchain", "learning", "llm",
        "main_modules", "memory", "multimodal", "planning", "plugins",
        "security", "system", "tools", "utils"
    ]
    
    print("\\nOld directories:")
    for dir_name in old_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  📁 {dir_name}/")
    
    # Check old files
    old_files = [
        "run.bat", "run-audio.bat", "run-mixed.bat", "run-text-only.bat",
        "run.sh", "install.bat", "install.sh", "install_ffmpeg.bat",
        "uninstall.sh", "check-system.sh", "dev.sh", "quickstart.sh",
        "test_ollama.py", "README.md", "CHANGELOG.md", "USAGE.md",
        "USAGE_GUIDE.md", ".env.example"
    ]
    
    print("\\nOld files:")
    for file_name in old_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"  📄 {file_name}")
    
    # Check temp files
    temp_files = ["reorganize_project.py", "create_init_files.py", "REORGANIZATION_COMPLETE.md"]
    
    print("\\nTemporary files:")
    for file_name in temp_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"  🗑️  {file_name}")
    
    # Check build artifacts
    if (base_dir / "mia_successor.egg-info").exists():
        print("\\nBuild artifacts:")
        print("  🏗️  mia_successor.egg-info/")
    
    print("\\n" + "=" * 50)

def verify_new_structure():
    """Verify that the new structure is intact"""
    base_dir = Path(".")
    
    required_files = [
        "main.py",
        "start.bat",
        "start-menu.bat",
        "src/mia/main.py",
        "src/mia/__init__.py",
        "requirements.txt",
        "setup.py",
        "LICENSE"
    ]
    
    required_dirs = [
        "src/mia",
        "scripts/install",
        "scripts/run",
        "scripts/development",
        "docs",
        "config"
    ]
    
    print("\\nVerifying new structure integrity...")
    print("-" * 40)
    
    all_good = True
    
    for file_path in required_files:
        if (base_dir / file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ MISSING: {file_path}")
            all_good = False
    
    for dir_path in required_dirs:
        if (base_dir / dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ MISSING: {dir_path}/")
            all_good = False
    
    if all_good:
        print("\\n✅ New structure is intact!")
    else:
        print("\\n❌ Some required files/directories are missing!")
        return False
    
    return True

def main():
    """Main cleanup function"""
    print("M.I.A Project Cleanup Tool")
    print("=" * 40)
    
    # Check what will be removed
    check_before_cleanup()
    
    # Ask for confirmation
    print("\\n⚠️  WARNING: This will permanently delete old files!")
    print("Make sure you have a backup if needed.")
    
    response = input("\\nProceed with cleanup? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        # Verify new structure first
        if verify_new_structure():
            print("\\n🧹 Starting cleanup...")
            cleanup_old_files()
            
            print("\\n" + "=" * 40)
            print("🎉 Cleanup completed successfully!")
            print("\\n📋 Summary:")
            print("  - Removed old duplicate directories")
            print("  - Removed old script files")
            print("  - Removed temporary files")
            print("  - Cleaned build artifacts")
            print("  - Preserved new organized structure")
            
            print("\\n✅ Your M.I.A project is now clean and organized!")
        else:
            print("\\n❌ Cleanup aborted - new structure verification failed!")
    else:
        print("\\n🚫 Cleanup cancelled.")

if __name__ == "__main__":
    main()
