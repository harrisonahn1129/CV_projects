#!/usr/bin/env python3
"""
Student Template for Constellation Classification
Replace 'netid' with your actual NetID (e.g., abc123.py)

REQUIREMENTS:
1. Command line: python netid.py <root_folder> -f <folder_name> [-v]
2. Output CSV: {folder_name}_results.csv saved in same directory as this script
3. CSV format: See specification below
"""

import pandas as pd
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import os
from PIL import Image
import networkx as nx
from sklearn.cluster import DBSCAN

def load_image(image_path):
    """
    Load image from various formats (.tif, .png, .jpg)
    Returns numpy array in BGR format for OpenCV
    """
    try:
        # Try loading with OpenCV first
        img = cv2.imread(str(image_path))
        if img is not None:
            return img
        
        # If OpenCV fails, try with PIL and convert
        with Image.open(image_path) as pil_img:
            # Convert to RGB if needed
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            # Convert PIL to numpy array and then to BGR for OpenCV
            img_array = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_bgr
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def template_matching(sky_image, patch_image, threshold=0.5):
    """
    Perform template matching between sky image and patch
    Returns (x, y) coordinates of best match or (-1, -1) if no good match
    """
    try:
        # Convert to grayscale for template matching
        sky_gray = cv2.cvtColor(sky_image, cv2.COLOR_BGR2GRAY)
        patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(sky_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Check if match is above threshold
        if max_val >= threshold:
            # Get center coordinates of the match
            h, w = patch_gray.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return (center_x, center_y)
        else:
            return (-1, -1)
            
    except Exception as e:
        print(f"Error in template matching: {e}")
        return (-1, -1)


def validate_patch_quality(sky_image, patch_image, match_coords, threshold=0.4):
    """
    Validate if a detected patch is of sufficient quality and belongs to the target constellation
    Returns True if patch is valid, False if it should be rejected
    Uses more permissive thresholds
    """
    try:
        if match_coords == (-1, -1):
            return False
        
        # Extract patch region from sky image for validation
        patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY) if len(patch_image.shape) > 2 else patch_image
        h, w = patch_gray.shape
        
        x, y = match_coords
        # Extract region around the match
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(sky_image.shape[1], x + w//2)
        y2 = min(sky_image.shape[0], y + h//2)
        
        if x2 - x1 < w//3 or y2 - y1 < h//3:  # More permissive size check
            return False
        
        sky_region = sky_image[y1:y2, x1:x2]
        if sky_region.size == 0:
            return False
        
        # Convert to grayscale for comparison
        sky_region_gray = cv2.cvtColor(sky_region, cv2.COLOR_BGR2GRAY) if len(sky_region.shape) > 2 else sky_region
        
        # Resize to match patch size
        sky_region_resized = cv2.resize(sky_region_gray, (w, h))
        
        # Calculate correlation for validation
        correlation = cv2.matchTemplate(sky_region_resized, patch_gray, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Additional validation: check for reasonable brightness and contrast
        mean_brightness = np.mean(sky_region_resized)
        std_brightness = np.std(sky_region_resized)
        
        # More permissive brightness/contrast checks
        if mean_brightness < 20 or std_brightness < 5:
            return False
        
        return correlation >= threshold
        
    except Exception as e:
        # Be more permissive with errors - default to accepting the patch
        return True

def find_patches_in_sky_image(sky_image_path, patches_folder, verbose=False):
    """
    Find all patches in the sky image using template matching with validation
    Returns dictionary with patch names and their coordinates
    """
    patch_results = {}
    
    # Load the sky image
    sky_image = load_image(sky_image_path)
    if sky_image is None:
        print(f"Failed to load sky image: {sky_image_path}")
        return patch_results
    
    if verbose:
        print(f"Loaded sky image: {sky_image.shape}")
    
    # Get all patch files
    patch_files = list(Path(patches_folder).glob("*.png"))
    patch_files.extend(list(Path(patches_folder).glob("*.jpg")))
    patch_files = sorted(patch_files)
    
    if verbose:
        print(f"Found {len(patch_files)} patches to match")
    
    valid_patches = []
    
    for patch_file in patch_files:
        patch_name = patch_file.name
        
        # Load patch image
        patch_image = load_image(patch_file)
        if patch_image is None:
            patch_results[patch_name] = (-1, -1)
            continue
        
        # Perform template matching
        x, y = template_matching(sky_image, patch_image)
        
        # Validate patch quality
        if x != -1 and validate_patch_quality(sky_image, patch_image, (x, y)):
            patch_results[patch_name] = (x, y)
            valid_patches.append((x, y))
            if verbose:
                print(f"Valid match for {patch_name} at ({x}, {y})")
        else:
            patch_results[patch_name] = (-1, -1)
            if verbose:
                print(f"Rejected patch {patch_name} - low quality or no match")
    
    # Advanced spatial clustering to remove outliers using DBSCAN
    if len(valid_patches) > 3:
        try:
            # Convert to numpy array for clustering
            points = np.array(valid_patches)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=150, min_samples=2).fit(points)
            labels = clustering.labels_
            
            # Find the largest cluster (excluding noise points labeled as -1)
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)  # Remove noise label
            
            if unique_labels:
                # Find the largest cluster
                largest_cluster = max(unique_labels, key=lambda x: np.sum(labels == x))
                
                # Reject patches not in the largest cluster
                for i, (patch_name, coords) in enumerate(patch_results.items()):
                    if coords != (-1, -1):
                        # Find which patch this corresponds to
                        patch_index = valid_patches.index(coords)
                        if labels[patch_index] != largest_cluster:
                            patch_results[patch_name] = (-1, -1)
                            if verbose:
                                print(f"Rejected outlier patch {patch_name} (not in main cluster)")
        except Exception as e:
            if verbose:
                print(f"Clustering failed, using fallback method: {e}")
            # Fallback to simple distance-based filtering
            center_x = sum(pos[0] for pos in valid_patches) / len(valid_patches)
            center_y = sum(pos[1] for pos in valid_patches) / len(valid_patches)
            distances = [np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2) for pos in valid_patches]
            max_distance = max(distances)
            outlier_threshold = max_distance * 0.8
            
            for patch_name, coords in patch_results.items():
                if coords != (-1, -1):
                    distance_from_center = np.sqrt((coords[0] - center_x)**2 + (coords[1] - center_y)**2)
                    if distance_from_center > outlier_threshold:
                        patch_results[patch_name] = (-1, -1)
                        if verbose:
                            print(f"Rejected outlier patch {patch_name} at distance {distance_from_center:.1f}")
    
    return patch_results

def analyze_patch_positions(patch_results):
    """
    Analyze the spatial distribution of detected patches
    Returns a list of (x, y) coordinates of successful matches
    """
    positions = []
    for patch_name, coords in patch_results.items():
        if coords != (-1, -1):
            x, y = coords
            positions.append((x, y))
    return positions


def create_constellation_graph(patches, max_distance=200):
    """
    Create a graph from detected patch coordinates
    """
    G = nx.Graph()
    
    if len(patches) < 2:
        return G
    
    # Add nodes with positions
    for i, (x, y) in enumerate(patches):
        G.add_node(i, pos=(x, y))
    
    # Add edges based on spatial proximity
    for i in range(len(patches)):
        for j in range(i+1, len(patches)):
            dist = np.sqrt((patches[i][0] - patches[j][0])**2 + 
                          (patches[i][1] - patches[j][1])**2)
            if dist <= max_distance:  # Only connect nearby patches
                G.add_edge(i, j, weight=dist)
    
    return G

def create_pattern_graph(pattern_image, max_distance=200):
    """
    Create a graph from constellation pattern image
    """
    try:
        # Load pattern image
        pattern = load_image(pattern_image)
        if pattern is None:
            return nx.Graph()
        
        # Convert to grayscale
        pattern_gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
        
        # Use ORB to detect keypoints in pattern
        orb = cv2.ORB_create(nfeatures=100)
        kp, des = orb.detectAndCompute(pattern_gray, None)
        
        if kp is None or len(kp) < 2:
            return nx.Graph()
        
        # Extract coordinates
        pattern_points = [(int(k.pt[0]), int(k.pt[1])) for k in kp]
        
        # Create graph
        G = nx.Graph()
        for i, (x, y) in enumerate(pattern_points):
            G.add_node(i, pos=(x, y))
        
        # Add edges based on spatial proximity
        for i in range(len(pattern_points)):
            for j in range(i+1, len(pattern_points)):
                dist = np.sqrt((pattern_points[i][0] - pattern_points[j][0])**2 + 
                              (pattern_points[i][1] - pattern_points[j][1])**2)
                if dist <= max_distance:
                    G.add_edge(i, j, weight=dist)
        
        return G
        
    except Exception as e:
        return nx.Graph()

def calculate_graph_similarity(G1, G2):
    """
    Calculate similarity between two graphs
    """
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return 0.0
    
    # Calculate various graph metrics
    metrics = {}
    
    # Node count similarity
    node_ratio = min(G1.number_of_nodes(), G2.number_of_nodes()) / max(G1.number_of_nodes(), G2.number_of_nodes())
    metrics['node_ratio'] = node_ratio
    
    # Edge count similarity
    edge_ratio = min(G1.number_of_edges(), G2.number_of_edges()) / max(G1.number_of_edges(), G2.number_of_edges()) if max(G1.number_of_edges(), G2.number_of_edges()) > 0 else 0
    metrics['edge_ratio'] = edge_ratio
    
    # Density similarity
    density1 = nx.density(G1)
    density2 = nx.density(G2)
    density_sim = 1 - abs(density1 - density2) / max(density1, density2) if max(density1, density2) > 0 else 0
    metrics['density_sim'] = density_sim
    
    # Average clustering coefficient similarity
    try:
        cc1 = nx.average_clustering(G1)
        cc2 = nx.average_clustering(G2)
        cc_sim = 1 - abs(cc1 - cc2) / max(cc1, cc2) if max(cc1, cc2) > 0 else 0
        metrics['cc_sim'] = cc_sim
    except:
        metrics['cc_sim'] = 0
    
    # Weighted average of all metrics
    similarity = (metrics['node_ratio'] * 0.3 + 
                  metrics['edge_ratio'] * 0.3 + 
                  metrics['density_sim'] * 0.2 + 
                  metrics['cc_sim'] * 0.2)
    
    return similarity

def calculate_pattern_similarity_graph(detected_positions, pattern_image, sky_image_size, verbose=False):
    """
    Calculate similarity between detected patch positions and constellation pattern using graph matching
    """
    try:
        if not detected_positions:
            return 0.0
        
        # Extract constellation name from pattern filename
        constellation_name = pattern_image.stem.replace("_pattern", "").lower()
        
        # Create graphs
        detected_graph = create_constellation_graph(detected_positions)
        pattern_graph = create_pattern_graph(pattern_image)
        
        if detected_graph.number_of_nodes() == 0 or pattern_graph.number_of_nodes() == 0:
            return 0.0
        
        # Calculate graph similarity
        graph_similarity = calculate_graph_similarity(detected_graph, pattern_graph)
        
        # Additional scoring based on constellation characteristics
        num_matches = len(detected_positions)
        
        # Match count bonus
        match_bonus = 0.0
        if 2 <= num_matches <= 8:
            match_bonus = 0.2
        elif 1 <= num_matches <= 12:
            match_bonus = 0.15
        elif 1 <= num_matches <= 20:
            match_bonus = 0.1
        
        # Final similarity score
        similarity = graph_similarity * 0.7 + match_bonus
        
        if verbose:
            print(f"Pattern: {constellation_name}, Patches: {num_matches}, Graph sim: {graph_similarity:.3f}, Final: {similarity:.3f}")
        
        return similarity
        
    except Exception as e:
        if verbose:
            print(f"Error calculating graph similarity for {pattern_image}: {e}")
        return 0.0


def calculate_constellation_similarity(patch_results, pattern_image, verbose=False):
    """
    Calculate similarity between detected patch positions and constellation pattern
    Returns a similarity score (0-1)
    """
    try:
        if not patch_results:
            return 0.0
        
        # Get detected positions
        detected_positions = []
        for patch_name, coords in patch_results.items():
            if coords != (-1, -1):
                detected_positions.append(coords)
        
        if not detected_positions:
            return 0.0
        
        # Estimate sky image size from detected positions
        max_x = max(pos[0] for pos in detected_positions)
        max_y = max(pos[1] for pos in detected_positions)
        sky_image_size = (max_y + 100, max_x + 100)  # Add some padding
        
        # Use graph-based pattern matching approach
        return calculate_pattern_similarity_graph(detected_positions, pattern_image, sky_image_size, verbose)
        
    except Exception as e:
        if verbose:
            print(f"Error calculating similarity for {pattern_image}: {e}")
        return 0.0

def classify_constellation(patch_results, patterns_folder, verbose=False):
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
        
        for pattern_file in pattern_files:
            similarity = calculate_constellation_similarity(patch_results, pattern_file, verbose)
            
            if similarity > best_score:
                best_score = similarity
                # Extract constellation name from filename
                constellation_name = pattern_file.stem.replace("_pattern", "").replace("-", "-")
                best_match = constellation_name
        
        # Only return a match if similarity is above threshold
        if best_score > 0.05:  # Even lower threshold for pattern matching
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
    
    # Sort by numerical order (constellation_1, constellation_2, etc.)
    def sort_key(folder_name):
        # Extract the number from constellation_X format
        try:
            return int(folder_name.name.split('_')[-1])
        except (ValueError, IndexError):
            return 0
    
    constellation_folders = sorted(constellation_folders, key=sort_key)
    
    if verbose:
        print(f"Found {len(constellation_folders)} constellation folders")
    
    # IMPLEMENTED ALGORITHM: Template matching for constellation detection
    all_results = []
    max_patches = 0  # Will be determined dynamically by scanning folders
    
    for i, constellation_folder in enumerate(constellation_folders, 1):
        folder_name = constellation_folder.name
        
        if verbose:
            print(f"\nProcessing {folder_name}...")
        
        # Find the sky image file
        sky_image_files = []
        for ext in ['*.tif', '*.png', '*.jpg', '*.jpeg']:
            sky_image_files.extend(list(constellation_folder.glob(ext)))
        
        if not sky_image_files:
            print(f"No sky image found in {constellation_folder}")
            continue
            
        sky_image_path = sky_image_files[0]  # Take the first image found
        
        # Find patches folder
        patches_folder = constellation_folder / "patches"
        if not patches_folder.exists():
            print(f"No patches folder found in {constellation_folder}")
            continue
        
        # Find patches in the sky image using template matching
        patch_results = find_patches_in_sky_image(sky_image_path, patches_folder, verbose)
        
        # Classify constellation based on patch matches
        constellation_prediction = classify_constellation(patch_results, patterns_path, verbose)
        
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