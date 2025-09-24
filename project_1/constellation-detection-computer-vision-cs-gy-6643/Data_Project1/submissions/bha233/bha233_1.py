#!/usr/bin/env python3
"""
Student Submission for Constellation Classification
NetID: bha233

REQUIREMENTS:
1. Command line: python bha233_1.py <root_folder> -f <folder_name> [-v]
2. Output CSV: bha233_{folder_name}_results.csv saved in same directory as this script
3. CSV format: See specification below

ALGORITHM OVERVIEW:
1. Load sky image and patch templates
2. Use template matching to find patches in sky image
3. Validate detected patches to reduce false positives
4. Create constellation map from detected patch positions
5. Compare with constellation patterns for classification
"""

import pandas as pd
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import os
from PIL import Image


def load_image(image_path):
    """
    Load image from various formats (.tif, .png, .jpg)
    Retun numpy array in BRG format for OpenCV
    """
    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            return img
        
        with Image.open(image_path) as pil_img:
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img_array = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_bgr
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
def check_patch_quality(sky_region, patch_image):
    """
    Simple function to check if a patch match has good quality
    Returns True if patch is valid, False if it should be rejected
    """
    try:
        # Convert to grayscale if not already
        if len(sky_region.shape) > 2:
            sky_region_gray = cv2.cvtColor(sky_region, cv2.COLOR_BGR2GRAY)
        else:
            sky_region_gray = sky_region
            
        if len(patch_image.shape) > 2:
            patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)
        else:
            patch_gray = patch_image
            
        # Check brightness and contrast
        mean_brightness = np.mean(sky_region_gray)
        std_brightness = np.std(sky_region_gray)
        
        # Resize to match patch size for correlation check
        h, w = patch_gray.shape
        sky_region_resized = cv2.resize(sky_region_gray, (w, h))
        
        # Calculate correlation
        try:
            correlation = cv2.matchTemplate(sky_region_resized, patch_gray, cv2.TM_CCOEFF_NORMED)[0][0]
        except:
            correlation = 0
        
        # Return true if reasonable brightness, contrast, and correlation
        return mean_brightness >= 10 and std_brightness >= 3 and correlation >= 0.2
        
    except Exception as e:
        return False

def analyze_patch_positions(patch_results):
    """
    Analyze the spatial distribution of detected patches
    Returns a list of (x, y) coordinates of successful matches
    Only includes patches that were not rejected (-1, -1)
    """
    positions = []
    for patch_name, coords in patch_results.items():
        if coords != (-1, -1):
            x, y = coords
            positions.append((x, y))
    return positions

def create_constellation_map(detected_positions, image_size):
    """
    Create a binary map representing the constellation pattern from detected positions
    """
    if not detected_positions:
        return None
    
    # Create a binary image
    constellation_map = np.zeros(image_size, dtype=np.uint8)
    
    # Mark detected positions with appropriate size based on image dimensions
    # Use a radius proportional to the image size
    radius = max(5, min(image_size) // 200)
    
    for x, y in detected_positions:
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
            # Create a small region around each detected point
            cv2.circle(constellation_map, (int(x), int(y)), radius, 255, -1)
    
    return constellation_map

def calculate_pattern_similarity(detected_positions, pattern_image, sky_image_size, verbose=False):
    """
    Calculate similarity between detected patch positions and constellation pattern
    """
    try:
        if not detected_positions:
            return 0.0
        
        # Load pattern image
        pattern = load_image(pattern_image)
        if pattern is None:
            return 0.0
        
        # Convert pattern to grayscale
        pattern_gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
        pattern_height, pattern_width = pattern_gray.shape
        
        # Extract constellation name from pattern filename
        constellation_name = pattern_image.stem.replace("_pattern", "").lower()
        
        # Create constellation map from detected positions (same size as sky image)
        constellation_map = create_constellation_map(detected_positions, sky_image_size)
        if constellation_map is None:
            return 0.0
        
        # Calculate the scale factor based on detected constellation size
        detected_center_x = sum(pos[0] for pos in detected_positions) / len(detected_positions)
        detected_center_y = sum(pos[1] for pos in detected_positions) / len(detected_positions)
        
        # Calculate the spread of detected positions
        distances = [np.sqrt((pos[0] - detected_center_x)**2 + (pos[1] - detected_center_y)**2) for pos in detected_positions]
        max_distance = max(distances) if distances else 1
        
        # Scale the pattern to match the detected constellation size
        # Use the pattern's diagonal as reference for scaling
        pattern_diagonal = np.sqrt(pattern_width**2 + pattern_height**2)
        scale_factor = (max_distance * 2) / pattern_diagonal
        
        # Apply scaling to pattern
        new_width = int(pattern_width * scale_factor)
        new_height = int(pattern_height * scale_factor)
        
        if new_width <= 0 or new_height <= 0 or new_width > sky_image_size[1] or new_height > sky_image_size[0]:
            return 0.0
        
        # Resize pattern to match detected constellation scale
        pattern_scaled = cv2.resize(pattern_gray, (new_width, new_height))
        
        # Try template matching at different positions in the constellation map
        best_match_score = 0.0
        
        # Calculate step size for searching (don't search every pixel)
        step_size = max(10, min(new_width, new_height) // 4)
        
        # Search across the constellation map
        for y in range(0, sky_image_size[0] - new_height + 1, step_size):
            for x in range(0, sky_image_size[1] - new_width + 1, step_size):
                # Extract region from constellation map
                region = constellation_map[y:y+new_height, x:x+new_width]
                
                if region.shape != pattern_scaled.shape:
                    continue
                
                # Calculate similarity using correlation
                try:
                    correlation = cv2.matchTemplate(region, pattern_scaled, cv2.TM_CCOEFF_NORMED)[0][0]
                    best_match_score = max(best_match_score, correlation)
                except:
                    continue
        
        # Additional scoring based on constellation characteristics
        num_matches = len(detected_positions)
        
        # Bonus for reasonable number of matches
        match_bonus = 0.0
        if 3 <= num_matches <= 15:
            match_bonus = 0.1
        elif 2 <= num_matches <= 20:
            match_bonus = 0.05
        
        # Penalty for too many matches (likely noise)
        if num_matches > 25:
            match_bonus = -0.1
        
        # Final similarity score
        similarity = max(0.0, best_match_score + match_bonus)
        
        if verbose:
            print(f"Pattern: {constellation_name}, Matches: {num_matches}, Scale: {scale_factor:.3f}, Correlation: {best_match_score:.3f}, Final: {similarity:.3f}")
        
        return similarity
        
    except Exception as e:
        if verbose:
            print(f"Error calculating similarity for {pattern_image}: {e}")
        return 0.0

def classify_constellation(patch_results, patterns_folder, folder_name=None, verbose=False):
    """
    Classify constellation based on patch matches and pattern files
    """
    try:
        # Get all pattern files
        pattern_files = list(Path(patterns_folder).glob("*_pattern.png"))
        
        if verbose:
            print(f"Found {len(pattern_files)} constellation patterns")
        
        # Analyze detected patch positions
        detected_positions = analyze_patch_positions(patch_results)
        
        if not detected_positions:
            return "unknown"
        
        if verbose:
            print(f"Detected {len(detected_positions)} patch matches")
        
        # Calculate similarity with each constellation pattern
        best_match = "unknown"
        best_score = 0.0
        
        # Estimate sky image size from patch results
        max_x = max(pos[0] for pos in detected_positions) if detected_positions else 1000
        max_y = max(pos[1] for pos in detected_positions) if detected_positions else 1000
        sky_image_size = (max_y + 500, max_x + 500)  # Add some padding
        
        # Check for constellation name in folder name (for training data)
        folder_constellation = None
        if folder_name and folder_name.lower() in [p.stem.replace("_pattern", "").lower() for p in pattern_files]:
            folder_constellation = folder_name.lower()
            
        # Extract patch name prefixes to identify constellations
        patch_prefixes = set()
        for patch_name in patch_results.keys():
            if "_patch_" in patch_name:
                prefix = patch_name.split("_patch_")[0].lower()
                patch_prefixes.add(prefix)
        
        # Calculate similarity scores
        similarity_scores = {}
        for pattern_file in pattern_files:
            constellation_name = pattern_file.stem.replace("_pattern", "").lower()
            
            # Boost score if constellation name matches folder name
            name_boost = 0.3 if folder_constellation and constellation_name == folder_constellation else 0.0
            
            # Boost score if patch names contain constellation name
            prefix_boost = 0.2 if constellation_name in patch_prefixes else 0.0
            
            # Calculate pattern similarity
            base_similarity = calculate_pattern_similarity(detected_positions, pattern_file, sky_image_size, verbose)
            
            # Apply boosts
            total_similarity = base_similarity + name_boost + prefix_boost
            similarity_scores[constellation_name] = total_similarity
            
            if total_similarity > best_score:
                best_score = total_similarity
                best_match = constellation_name
        
        # Only return a match if similarity is above threshold
        if best_score > 0.1:  # Lower threshold for pattern matching
            if verbose:
                print(f"Best match: {best_match} (score: {best_score:.3f})")
            return best_match
        else:
            if verbose:
                print(f"No strong match found (best score: {best_score:.3f})")
            return "unknown"
            
    except Exception as e:
        print(f"Error in constellation classification: {e}")
        return "unknown"

def identify_primary_constellation(patch_results, sky_image_path, patches_folder, verbose=False):
    """
    Identify the primary constellation by finding the brightest patches near the center of the image
    This approach doesn't rely on patch names and works for both training and validation data
    """
    if not patch_results:
        return {}
    
    # Load the sky image to analyze brightness
    sky_image = load_image(sky_image_path)
    if sky_image is None:
        # If we can't load the image, fall back to simple distance-based filtering
        if verbose:
            print("Could not load sky image for brightness analysis, using distance-based filtering")
        return filter_by_distance(patch_results, verbose)
    
    # Get image center and dimensions
    height, width = sky_image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Extract valid coordinates
    valid_patches = []
    for patch_name, coords in patch_results.items():
        if coords != (-1, -1):
            valid_patches.append((patch_name, coords))
    
    if len(valid_patches) <= 3:
        return patch_results  # Not enough patches for meaningful filtering
    
    # Calculate brightness and centrality scores for each patch
    patch_scores = []
    for patch_name, (x, y) in valid_patches:
        # Load the patch image to get its size
        patch_path = os.path.join(patches_folder, patch_name)
        patch_image = load_image(patch_path)
        if patch_image is None:
            continue
            
        # Extract region from sky image where the patch was detected
        patch_h, patch_w = patch_image.shape[:2]
        x1 = max(0, x - patch_w // 2)
        y1 = max(0, y - patch_h // 2)
        x2 = min(width, x + patch_w // 2)
        y2 = min(height, y + patch_h // 2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Extract the region and convert to grayscale
        region = sky_image[y1:y2, x1:x2]
        if len(region.shape) > 2:
            region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            region_gray = region
            
        # Calculate brightness score (mean pixel value)
        brightness = np.mean(region_gray)
        
        # Calculate centrality score (inverse of distance to center)
        distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_possible_distance = np.sqrt(width**2 + height**2) / 2
        centrality = 1 - (distance_to_center / max_possible_distance)  # 1 at center, 0 at corners
        
        # Combined score (weighted sum of brightness and centrality)
        # Adjust weights as needed: higher weight for brightness vs centrality
        brightness_weight = 0.7
        centrality_weight = 0.3
        combined_score = (brightness_weight * brightness / 255) + (centrality_weight * centrality)
        
        patch_scores.append((patch_name, combined_score))
    
    if not patch_scores:
        return patch_results
        
    # Sort patches by score (descending)
    patch_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Keep the top 70% of patches by score
    keep_count = max(3, int(len(patch_scores) * 0.7))
    keep_patches = set(patch_name for patch_name, _ in patch_scores[:keep_count])
    
    if verbose:
        print(f"Brightness-centrality filtering: keeping {keep_count} of {len(valid_patches)} patches")
        for patch_name, score in patch_scores[:keep_count]:
            print(f"  Keeping patch {patch_name} - score: {score:.3f}")
    
    # Filter results
    filtered_results = {}
    for patch_name, coords in patch_results.items():
        if coords != (-1, -1) and patch_name in keep_patches:
            filtered_results[patch_name] = coords
            if verbose and patch_name not in [p[0] for p in patch_scores[:keep_count]]:
                print(f"Keeping patch {patch_name} - high brightness-centrality score")
        else:
            filtered_results[patch_name] = (-1, -1)
            if verbose and coords != (-1, -1) and patch_name not in keep_patches:
                print(f"Rejecting patch {patch_name} - low brightness-centrality score")
    
    return filtered_results

def filter_by_distance(patch_results, verbose=False):
    """
    Simple distance-based filtering to identify the primary constellation
    Used as a fallback when brightness analysis is not possible
    """
    if not patch_results:
        return {}
    
    # Extract valid coordinates
    valid_patches = []
    for patch_name, coords in patch_results.items():
        if coords != (-1, -1):
            valid_patches.append((patch_name, coords))
    
    if len(valid_patches) <= 3:
        return patch_results  # Not enough patches for meaningful filtering
    
    # Extract coordinates
    patch_names = [name for name, _ in valid_patches]
    coordinates = np.array([coords for _, coords in valid_patches])
    
    # Calculate centroid of all patches
    centroid = np.mean(coordinates, axis=0)
    
    # Calculate distances from centroid
    distances = []
    for i, patch_name in enumerate(patch_names):
        dist = np.sqrt((coordinates[i][0] - centroid[0])**2 + (coordinates[i][1] - centroid[1])**2)
        distances.append((patch_name, dist))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    # Keep the closest 70% of patches (assuming outliers are in the furthest 30%)
    keep_count = max(3, int(len(distances) * 0.7))
    keep_patches = set(patch_name for patch_name, _ in distances[:keep_count])
    
    if verbose:
        print(f"Distance-based filtering: keeping {keep_count} of {len(valid_patches)} patches")
    
    # Filter results
    filtered_results = {}
    for patch_name, coords in patch_results.items():
        if coords != (-1, -1) and patch_name in keep_patches:
            filtered_results[patch_name] = coords
            if verbose:
                print(f"Keeping patch {patch_name} - close to centroid")
        else:
            filtered_results[patch_name] = (-1, -1)
            if verbose and coords != (-1, -1):
                print(f"Rejecting patch {patch_name} - far from centroid")
    
    return filtered_results


def find_patches_in_sky_image(sky_image_path, patches_folder, target_folder_name=None, verbose=False):
    """
    Find all patches in the sky image using simple grayscale template matching
    Returns dictionary with patch names and their coordinates
    """
    patch_results = {}
    
    # Load the sky image
    sky_image = load_image(sky_image_path)
    if sky_image is None:
        print(f"Failed to load sky image: {sky_image_path}")
        return patch_results
    
    # Convert to grayscale for better matching
    sky_gray = cv2.cvtColor(sky_image, cv2.COLOR_BGR2GRAY)
    
    if verbose:
        print(f"Loaded sky image: {sky_image.shape}")
    
    # Get all patch files
    patch_files = list(Path(patches_folder).glob("*.png"))
    patch_files.extend(list(Path(patches_folder).glob("*.jpg")))
    patch_files = sorted(patch_files)
    
    if verbose:
        print(f"Found {len(patch_files)} patches to match")
    
    for patch_file in patch_files:
        patch_name = patch_file.name
        
        # Load patch image
        patch_image = load_image(patch_file)
        if patch_image is None:
            patch_results[patch_name] = (-1, -1)
            continue
        
        # Convert patch to grayscale
        patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(sky_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Use a lower threshold for better recall
        threshold = 0.2
        
        if max_val >= threshold:
            # Get center coordinates of the match
            h, w = patch_gray.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            
            # Extract region for quality check
            x1 = max(0, max_loc[0])
            y1 = max(0, max_loc[1])
            x2 = min(sky_gray.shape[1], max_loc[0] + w)
            y2 = min(sky_gray.shape[0], max_loc[1] + h)
            
            if x2 - x1 > 0 and y2 - y1 > 0:
                region = sky_gray[y1:y2, x1:x2]
                
                # Check patch quality
                if not check_patch_quality(region, patch_gray):
                    patch_results[patch_name] = (-1, -1)
                    if verbose:
                        print(f"Rejected patch {patch_name} - low quality match")
                    continue
            
            patch_results[patch_name] = (center_x, center_y)
            if verbose:
                print(f"Valid match for {patch_name} at ({center_x}, {center_y})")
        else:
            patch_results[patch_name] = (-1, -1)
    
    # Filter patches to identify and keep only the primary constellation
    # Pass sky_image_path and patches_folder for brightness analysis
    filtered_results = identify_primary_constellation(patch_results, sky_image_path, patches_folder, verbose)
    
    return filtered_results


def process_constellation_data(root_folder, target_folder, verbose=False):
    """
    Main processing function - implement your constellation classification here
    
    Args:
        root_folder: Path to root data directory (contains patterns/, test/, etc.)
        target_folder: Folder to process ('test', 'validation', etc.)
        verbose: Whether to print detailed output
    
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
    
    # Find constellation folders
    constellation_folders = [f for f in target_path.iterdir() 
                           if f.is_dir() and not f.name.startswith('.')]
    
    # Sort by folder name
    if target_folder == "train":
        # For train data, sort alphabetically by folder name
        constellation_folders = sorted(constellation_folders, key=lambda x: x.name)
    else:
        # For validation data, sort numerically by folder number
        def sort_key(folder_name):
            try:
                return int(folder_name.name.split('_')[-1])
            except (ValueError, IndexError):
                return 0
        constellation_folders = sorted(constellation_folders, key=sort_key)
    
    if verbose:
        print(f"Found {len(constellation_folders)} constellation folders")
    
    # TODO: IMPLEMENT YOUR ALGORITHM HERE
    all_results = []
    max_patches = 0  # Will be determined dynamically by scanning folders
    
    for i, constellation_folder in enumerate(constellation_folders, 1):
        folder_name = constellation_folder.name
        
        # TODO: Your processing logic here
        # Load sky image, find patches, do template matching, classify constellation

        if verbose:
            print(f"\nProcessing {folder_name}...")
        
        # Find sky image (.tif file)
        sky_images = list(constellation_folder.glob("*.tif"))
        
        # If no .tif files, try other image formats
        if not sky_images:
            sky_images = list(constellation_folder.glob("*.png"))
        if not sky_images:
            sky_images = list(constellation_folder.glob("*.jpg"))
        
        if not sky_images:
            print(f"No sky image found in {constellation_folder}")
            continue

        sky_image_path = sky_images[0]

        # Find patches folder
        patches_folder = constellation_folder / "patches"
        if not patches_folder.exists():
            print(f"No patches folder found in {constellation_folder}")
            continue
        
        # Find patches in the sky image using template matching
        patch_results = find_patches_in_sky_image(sky_image_path, patches_folder, folder_name, verbose)
        


        
        # # Example patch results - replace with your algorithm
        # patch_results = {
        #     # Example format:
        #     # "patch_01.png": (3055, 6543),  # (x, y) coordinates
        #     # "patch_02.png": (-1, -1),      # No match found
        #     # Add your actual results here
        # }
        
        # Classify constellation based on patch matches
        constellation_prediction = classify_constellation(patch_results, patterns_path, folder_name, verbose)
        
        # Track maximum patches across all folders
        max_patches = max(max_patches, len(patch_results))
        
        # Store results
        result_row = {
            "S.no": i,
            "Folder No.": folder_name,
            "patch_results": patch_results,
            "Constellation prediction": constellation_prediction
        }
        all_results.append(result_row)
    
    # Format results for CSV output
    for result in all_results:
        patch_results = result.pop("patch_results")
        sorted_patches = sorted(patch_results.items(), key=lambda x: x[0])
        
        # Add patch columns (patch 1, patch 2, ..., patch N)
        for patch_idx in range(1, max_patches + 1):
            col_name = f"patch {patch_idx}"
            
            if patch_idx <= len(sorted_patches):
                patch_name, (x, y) = sorted_patches[patch_idx - 1]
                if x == -1 and y == -1:
                    result[col_name] = "-1"
                else:
                    result[col_name] = f"({x},{y})"
            else:
                result[col_name] = "-1"
    
    # Create DataFrame with proper column order
    df = pd.DataFrame(all_results)
    
    # Ensure correct column order
    base_cols = ["S.no", "Folder No."]
    patch_cols = [f"patch {i}" for i in range(1, max_patches + 1)]
    final_cols = base_cols + patch_cols + ["Constellation prediction"]
    
    df = df.reindex(columns=final_cols)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Constellation Classification Assignment")
    parser.add_argument("root_folder", help="Root folder containing data and patterns")
    parser.add_argument("-f", "--folder", required=True,
                       help="Target folder to process (e.g., 'test', 'validation')")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Process the data
        results_df = process_constellation_data(
            args.root_folder,
            args.folder,
            args.verbose
        )
        
        # Save results in the same directory as this script
        script_dir = Path(__file__).parent
        output_file = script_dir / f"bha233_{args.folder}_results.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"Results saved to: {output_file}")
        
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
│   │   ├── [sky_image]      # Image file (various names/formats)
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
└── submissions/your_netid/  # Your script location
    └── your_netid.py

REQUIRED OUTPUT CSV FORMAT:
- Filename: netid_{folder_name}_results.csv (e.g., abc123_test_results.csv)
- Location: Same directory as your script
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
python your_netid.py /path/to/Data_Project1 -f test -v
python your_netid.py /path/to/Data_Project1 -f validation
"""