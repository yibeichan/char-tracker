import os
import sys
import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from dotenv import load_dotenv

def read_json(file_path):
    """Read JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        return json.load(f)

def reorganize_by_cluster(video_name, matched_faces_file, source_dir, output_base_dir, mode='copy'):
    """
    Reorganize face images by cluster ID.

    Args:
        video_name: Episode ID (e.g., 'friends_s01e05')
        matched_faces_file: Path to the matched_faces_with_clusters.json
        source_dir: Directory containing the original flat image structure
        output_base_dir: Base directory for reorganized output
        mode: 'copy', 'move', or 'symlink'
    """
    matched_faces = read_json(matched_faces_file)

    # Create output directory for this episode
    episode_output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(episode_output_dir, exist_ok=True)

    # Collect all images by cluster
    cluster_images = defaultdict(list)

    for scene_id, faces in matched_faces.items():
        for face_data in faces:
            cluster_id = face_data.get('cluster_id')
            unique_face_id = face_data.get('unique_face_id')
            image_paths = face_data.get('image_paths', [])

            # Skip faces without cluster assignment
            if cluster_id is None:
                print(f"Warning: {unique_face_id} has no cluster assignment. Skipping.")
                continue

            # Extract scene and track info from unique_face_id (format: "scene_X_face_Y")
            parts = unique_face_id.split('_')
            if len(parts) >= 4 and parts[0] == 'scene' and parts[2] == 'face':
                scene_num = parts[1]
                track_num = parts[3]
            else:
                print(f"Warning: Unexpected unique_face_id format: {unique_face_id}")
                scene_num = scene_id
                track_num = "unknown"

            # Add each image to the cluster
            for img_path in image_paths:
                # Extract frame number from filename (format: "scene_X_face_Y_frame_ZZZ.jpg")
                filename = os.path.basename(img_path)
                frame_parts = os.path.splitext(filename)[0].split('_')
                if 'frame' in frame_parts:
                    frame_idx = frame_parts[frame_parts.index('frame') + 1]
                else:
                    frame_idx = "unknown"

                cluster_images[cluster_id].append({
                    'source_path': os.path.join(source_dir, img_path),
                    'scene': scene_num,
                    'track': track_num,
                    'frame': frame_idx
                })

    # Create cluster directories and organize files
    total_files = sum(len(images) for images in cluster_images.values())
    processed = 0

    print(f"Found {len(cluster_images)} clusters with {total_files} total images")

    for cluster_id, images in sorted(cluster_images.items()):
        # Format cluster directory name
        cluster_dir = os.path.join(episode_output_dir, f"{video_name}_cluster-{cluster_id:02d}")
        os.makedirs(cluster_dir, exist_ok=True)

        for img_info in images:
            source_path = img_info['source_path']

            # Check if source file exists
            if not os.path.exists(source_path):
                print(f"Warning: Source file not found: {source_path}")
                continue

            # Create new filename: scene_X_track_Y_frame_ZZZ.jpg
            new_filename = f"scene_{img_info['scene']}_track_{img_info['track']}_frame_{img_info['frame']}.jpg"
            dest_path = os.path.join(cluster_dir, new_filename)

            # Perform file operation based on mode
            try:
                if mode == 'copy':
                    shutil.copy2(source_path, dest_path)
                elif mode == 'move':
                    shutil.move(source_path, dest_path)
                elif mode == 'symlink':
                    # Create relative symlink, overwriting if it exists
                    rel_source = os.path.relpath(source_path, cluster_dir)
                    if os.path.lexists(dest_path):
                        os.remove(dest_path)
                    os.symlink(rel_source, dest_path)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                processed += 1
            except Exception as e:
                print(f"Error processing {source_path}: {e}")

        print(f"Created cluster directory: {cluster_dir} ({len(images)} images)")

    print(f"\nReorganization complete! Processed {processed}/{total_files} files")
    print(f"Output directory: {episode_output_dir}")

    return episode_output_dir

def create_zip_file(episode_output_dir, video_name):
    """
    Create a zip file containing all cluster directories.

    Args:
        episode_output_dir: Directory containing the cluster subdirectories
        video_name: Episode ID (e.g., 'friends_s01e01a')

    Returns:
        Path to the created zip file
    """
    # Create zip file in the same parent directory as episode_output_dir
    zip_path = os.path.join(os.path.dirname(episode_output_dir), f"{video_name}.zip")

    print(f"\nCreating zip file: {zip_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the episode directory
        for root, dirs, files in os.walk(episode_output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the archive name (relative path from episode_output_dir)
                # This puts cluster folders directly at the zip root
                arcname = os.path.relpath(file_path, episode_output_dir)
                zipf.write(file_path, arcname)

        # Get list of cluster directories for summary
        cluster_dirs = [d for d in os.listdir(episode_output_dir)
                       if os.path.isdir(os.path.join(episode_output_dir, d))]

    zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # Size in MB
    print(f"Zip file created successfully!")
    print(f"  - Path: {zip_path}")
    print(f"  - Size: {zip_size:.2f} MB")
    print(f"  - Contains {len(cluster_dirs)} cluster directories")

    return zip_path

def main(video_name, scratch_dir, mode, create_zip):
    matched_faces_file = os.path.join(scratch_dir, "output", "face_clustering",
                                      f"{video_name}_matched_faces_with_clusters.json")
    source_dir = os.path.join(scratch_dir, "output", "face_tracking", video_name)
    output_base_dir = os.path.join(scratch_dir, "output", "face_tracking_by_cluster")

    episode_output_dir = reorganize_by_cluster(video_name, matched_faces_file, source_dir, output_base_dir, mode)

    if create_zip:
        create_zip_file(episode_output_dir, video_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reorganize face images by cluster ID')
    parser.add_argument('video_name', type=str,
                       help='Name of the episode (e.g., friends_s01e05)')
    parser.add_argument('--mode', type=str, default='copy',
                       choices=['copy', 'move', 'symlink'],
                       help='How to organize files: copy (default), move, or symlink')
    parser.add_argument('--create-zip', action='store_true',
                       help='Create a zip file of the reorganized cluster structure')

    args = parser.parse_args()
    video_name = args.video_name

    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")

    if not scratch_dir:
        print("Error: SCRATCH_DIR not found in environment")
        sys.exit(1)

    main(video_name, scratch_dir, args.mode, args.create_zip)
