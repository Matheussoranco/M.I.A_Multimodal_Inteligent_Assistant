import os
import glob

# Create __init__.py files for all directories in src/mia
for dir_path in glob.glob('src/mia/*/'):
    init_file = os.path.join(dir_path, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""M.I.A module"""\n')
        print(f"Created: {init_file}")
    else:
        print(f"Exists: {init_file}")
