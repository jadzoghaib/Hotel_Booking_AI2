import subprocess, sys, os, zipfile, glob

# Install required packages
packages = ['imbalanced-learn', 'xgboost', 'lightgbm', 'catboost', 'nbformat', 'nbconvert', 'ipykernel', 'jupyter']
for pkg in packages:
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=False)

print("Setup complete!")