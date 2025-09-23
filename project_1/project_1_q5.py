#!/usr/bin/env python3
"""
Student Template for Constellation Classification with Random Simulation
Name this file with your NetID (e.g., abc123.py)

REQUIREMENTS:
1. Command line: python abc123.py <root_folder> [target_folder]
2. Output CSV: {script_name}_{folder_name}_results.csv saved in same directory as this script
3. CSV format: See specification below

ALGORITHM SECTIONS TO REPLACE:
- Template matching section: Replace random simulation with actual template matching
- Constellation classification: Replace random choice with actual pattern analysis
- Keep all CSV formatting code unchanged
"""

import pandas as pd
import argparse
import sys
import random
import os
import re
from pathlib import Path
from PIL import Image

def natural_sort_key(folder_path):
    """
    Generate a sort key for natural (numeric) sorting of folder names.
    E.g., constellation_1, constellation_2, ..., constellation_10, constellation_11
    """
    folder_name = folder_path.name
    # Extract numbers from the folder name and convert to integers for proper sorting
    numbers = re.findall(r'\d+', folder_name)
    # Create a key that sorts alphabetic parts normally and numeric parts as integers
    parts = re.split(r'(\d+)', folder_name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key

def get_image_dimensions(image_path):
    """Get dimensions of an image file"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return (1000, 1000)  # Default fallback dimensions

def get_constellation_names_from_patterns(patterns_path):
    """Extract constellation names from pattern files"""
    constellation_names = []
    if patterns_path.exists():
        for pattern_file in patterns_path.iterdir():
            if pattern_file.is_file() and pattern_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Remove '_pattern' suffix and file extension
                name = pattern_file.stem
                if name.endswith('_pattern'):
                    name = name[:-8]  # Remove '_pattern'
                constellation_names.append(name.lower())
    
    # Add some default constellation names if none found
    if not constellation_names:
        constellation_names = [
            'orion', 'bootes', 'cassiopeia', 'ursa-major', 'ursa-minor',
            'corona-australis', 'leo', 'virgo', 'scorpius', 'unknown'
        ]
    
    return constellation_names

def find_sky_image(constellation_folder):
    """Find the sky image in a constellation folder (not in patches subfolder)"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    
    for file in constellation_folder.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            # Make sure it's not in the patches subfolder
            if file.parent.name != 'patches':
                return file
    
    return None

def get_patch_files(patches_folder):
    """Get all patch files from the patches folder"""
    if not patches_folder.exists():
        return []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    patch_files = []
    
    for file in patches_folder.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            patch_files.append(file)
    
    return sorted(patch_files, key=lambda x: x.name)

def process_constellation_data(root_folder, target_folder, verbose=True):
    """
    Main processing function - implement your constellation classification here
    
    WHAT TO REPLACE:
    - Template matching algorithm (currently random simulation)
    - Constellation classification (currently random choice)
    
    WHAT TO KEEP:
    - All file loading and folder scanning code
    - CSV formatting and output generation
    - Command line argument handling
    
    Args:
        root_folder: Path to root data directory (contains patterns/, test/, etc.)
        target_folder: Folder to process ('test', 'validation', etc.)
        verbose: Whether to print detailed output (default: True)
    
    Returns:
        pandas.DataFrame: Results in required CSV format
    """
    
    root_path = Path(root_folder)
    target_path = root_path / target_folder
    patterns_path = root_path / "patterns"
    
    if verbose:
        print(f"Root folder: {root_path}")
        print(f"Target folder: {target_path}")
        print(f"Patterns folder: {patterns_path}")
    
    # Get constellation names from patterns
    constellation_names = get_constellation_names_from_patterns(patterns_path)
    if verbose:
        print(f"Available constellations: {constellation_names}")
    
    # Find constellation folders
    constellation_folders = [f for f in target_path.iterdir() 
                           if f.is_dir() and not f.name.startswith('.')]
    constellation_folders = sorted(constellation_folders, key=natural_sort_key)
    
    if verbose:
        print(f"Found {len(constellation_folders)} constellation folders")
    
    # Set random seed for reproducible results (optional)
    random.seed(42)
    
    all_results = []
    max_patches = 0  # Will be determined dynamically by scanning folders
    
    # First pass: determine maximum number of patches
    for constellation_folder in constellation_folders:
        patches_folder = constellation_folder / "patches"
        patch_files = get_patch_files(patches_folder)
        max_patches = max(max_patches, len(patch_files))
    
    if verbose:
        print(f"Maximum patches found in any folder: {max_patches}")
    
    for i, constellation_folder in enumerate(constellation_folders, 1):
        folder_name = constellation_folder.name
        
        if verbose:
            print(f"\nProcessing folder {i}: {folder_name}")
        
        # Find sky image
        sky_image_path = find_sky_image(constellation_folder)
        if sky_image_path is None:
            if verbose:
                print(f"Warning: No sky image found in {constellation_folder}")
            image_width, image_height = 1000, 1000  # Default dimensions
        else:
            image_width, image_height = get_image_dimensions(sky_image_path)
            if verbose:
                print(f"Sky image: {sky_image_path.name} ({image_width}x{image_height})")
        
        # Get patches
        patches_folder = constellation_folder / "patches"
        patch_files = get_patch_files(patches_folder)
        
        if verbose:
            print(f"Found {len(patch_files)} patch files")
        
        # =================================================================
        # TODO: REPLACE THIS SECTION WITH YOUR TEMPLATE MATCHING ALGORITHM
        # =================================================================
        # This is just a simulation! Replace with your actual algorithm:
        # 1. Load and process the sky image
        # 2. For each patch, do template matching against the sky image
        # 3. Apply thresholding to determine if match is good enough
        # 4. Store coordinates (x,y) of best matches or (-1,-1) for no match
        #
        # EXPECTED OUTPUT FORMAT:
        # patch_results = {
        #     "patch_01.png": (x, y),     # Found at coordinates (x, y)
        #     "patch_02.png": (-1, -1),   # Not found / rejected
        #     "patch_03.png": (x2, y2),   # Found at coordinates (x2, y2)
        #     ...
        # }
        
        # Process each patch with 20% acceptance rate (REPLACE THIS!)
        patch_results = {}
        acceptance_rate = 0.2  # 20% chance of finding a match (REMOVE THIS!)
        
        for patch_file in patch_files:
            # SIMULATION CODE - REPLACE WITH YOUR TEMPLATE MATCHING:
            if random.random() < acceptance_rate:
                # Generate random coordinates within image bounds (REPLACE THIS!)
                # Assume patch is small, so give some margin from edges
                margin = 50
                x = random.randint(margin, max(margin + 1, image_width - margin))
                y = random.randint(margin, max(margin + 1, image_height - margin))
                patch_results[patch_file.name] = (x, y)
                
                if verbose:
                    print(f"  {patch_file.name}: FOUND at ({x}, {y})")
            else:
                # No match found
                patch_results[patch_file.name] = (-1, -1)
                
                if verbose:
                    print(f"  {patch_file.name}: NOT FOUND")
        
        # =================================================================
        # TODO: REPLACE THIS SECTION WITH YOUR CONSTELLATION CLASSIFICATION
        # =================================================================
        # This is just random! Replace with your actual algorithm:
        # 1. Analyze the found patch positions
        # 2. Compare against known constellation patterns from patterns/ folder
        # 3. Use pattern matching, geometric analysis, or ML to classify
        # 4. Return the best matching constellation name (lowercase)
        
        # Random constellation prediction (REPLACE THIS!)
        constellation_prediction = random.choice(constellation_names)
        
        # =================================================================
        # END OF ALGORITHM SECTIONS - KEEP EVERYTHING BELOW AS-IS
        # =================================================================
        # The following code handles CSV formatting and should NOT be changed
        
        if verbose:
            print(f"Constellation prediction: {constellation_prediction}")
        
        # Store results
        result_row = {
            "S.no": i,
            "Folder No.": folder_name,
            "patch_results": patch_results,
            "Constellation prediction": constellation_prediction
        }
        all_results.append(result_row)
    
    # Format results for CSV output with proper padding
    for result in all_results:
        patch_results = result.pop("patch_results")
        sorted_patches = sorted(patch_results.items(), key=lambda x: x[0])
        
        # Add patch columns (patch 1, patch 2, ..., patch max_patches)
        for patch_idx in range(1, max_patches + 1):
            col_name = f"patch {patch_idx}"
            
            if patch_idx <= len(sorted_patches):
                patch_name, (x, y) = sorted_patches[patch_idx - 1]
                if x == -1 and y == -1:
                    result[col_name] = "-1"
                else:
                    result[col_name] = f"({x},{y})"
            else:
                # Padding with -1 for folders with fewer patches
                result[col_name] = "-1"
    
    # Create DataFrame with proper column order
    df = pd.DataFrame(all_results)
    
    # Ensure correct column order
    base_cols = ["S.no", "Folder No."]
    patch_cols = [f"patch {i}" for i in range(1, max_patches + 1)]
    final_cols = base_cols + patch_cols + ["Constellation prediction"]
    
    df = df.reindex(columns=final_cols)
    
    if verbose:
        print(f"\nFinal results shape: {df.shape}")
        print("Sample output:")
        print(df.head())
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Constellation Classification Assignment")
    parser.add_argument("root_folder", help="Root folder containing data and patterns")
    parser.add_argument("target_folder", nargs='?', default="test", 
                       help="Target folder to process (default: 'test')")
    
    args = parser.parse_args()
    
    try:
        # Validate paths
        root_path = Path(args.root_folder)
        if not root_path.exists():
            raise FileNotFoundError(f"Root folder not found: {root_path}")
        
        target_path = root_path / args.target_folder
        if not target_path.exists():
            raise FileNotFoundError(f"Target folder not found: {target_path}")
        
        # Process the data (always verbose)
        results_df = process_constellation_data(
            args.root_folder,
            args.target_folder,
            verbose=True
        )
        
        # Save results in the same directory as this script
        script_dir = Path(__file__).parent
        script_name = Path(__file__).stem  # Get filename without extension (the netid)
        output_file = script_dir / f"{script_name}_{args.target_folder}_results.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"Results saved to: {output_file}")
        print(f"Output file will be: {script_name}_{args.target_folder}_results.csv")
        print(f"Output file size: {output_file.stat().st_size} bytes")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


# ==============================================================================
# INPUT/OUTPUT SPECIFICATION
# ==============================================================================

"""
INPUT STRUCTURE:
root_folder/
├── test/                    # Your target folder (or validation/, etc.)
│   ├── constellation_1/
│   │   ├── [sky_image]      # Image file
│   │   └── patches/         # Subfolder with patch templates
│   │       ├── patch_01.png
│   │       ├── patch_02.png
│   │       └── ...
│   ├── constellation_2/
│   │   └── ...
│   └── ...
├── patterns/                # Reference constellation patterns
│   ├── constellation_name_pattern.png
│   └── ...
└── submissions/abc123/       # Your script location
    └── abc123.py

REQUIRED OUTPUT CSV FORMAT:
- Filename: {script_name}_{folder_name}_results.csv (e.g., abc123_test_results.csv)
- Location: Same directory as your script
- Note: Script filename should be your NetID (e.g., abc123.py)
- Columns: S.no, Folder No., patch 1, patch 2, ..., patch N, Constellation prediction

EXAMPLE OUTPUT:
S.no,Folder No.,patch 1,patch 2,patch 3,patch 4,...,patch N,Constellation prediction
1,constellation_1,(3055,6543),(3895,4611),(4463,4661),-1,...,-1,bootes
2,constellation_2,-1,(2456,3789),-1,-1,...,-1,orion
3,constellation_3,-1,-1,-1,-1,...,-1,unknown

(Note: N = maximum number of patches found across all folders)

COORDINATE FORMAT:
- Successful match: (x,y) - center coordinates of matched patch
- No match/rejected: -1
- Number of patch columns: Dynamically determined by folder with most patches

CONSTELLATION NAMES:
- Lowercase format (e.g., 'bootes', 'orion', 'corona-australis')
- Extract from pattern filenames (remove '_pattern' suffix)

USAGE:
python abc123.py /path/to/Data_Project1 test
python abc123.py /path/to/Data_Project1 validation
python abc123.py /path/to/Data_Project1  # defaults to 'test' folder

TEMPLATE SIMULATION FEATURES (TO BE REPLACED):
- 20% acceptance rate for patch matching (replace with real template matching)
- Random coordinate generation within image bounds (replace with actual coordinates)
- Automatic -1 padding for folders with fewer patches (KEEP THIS)
- Random constellation prediction from available patterns (replace with real classification)
- Works with any data following the specified folder structure (KEEP THIS)
- Always shows verbose output for debugging (KEEP THIS)
"""