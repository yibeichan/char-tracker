import os
import json
import pandas as pd
from natsort import natsorted
from dotenv import load_dotenv

# Opt-in to the future behavior
pd.set_option('future.no_silent_downcasting', True)

def read_json_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_file_path(base_dir, *paths):
    return os.path.join(base_dir, *paths)

def process_episode(episode_id, scratch_dir, save_dir):
    try:
        tracked_file = generate_file_path(scratch_dir, "output", "03_face_tracking", episode_id, f"{episode_id}_selected_frames_per_face.json")
        cluster_file = generate_file_path(scratch_dir, "output", "04_face_clustering", f"{episode_id}_matched_faces_with_clusters.json")
        matched_file = generate_file_path(scratch_dir, "output", "06_cluster_face_matching", f"{episode_id}_cluster-face_matching.json")
        scene_detect_file = generate_file_path(scratch_dir, "output", "01_scene_detection", f"{episode_id}.txt")

        tracked_face = read_json_file(tracked_file)
        clustered_face = read_json_file(cluster_file)
        matched_face = read_json_file(matched_file)
        scene_detect = pd.read_csv(scene_detect_file, sep=",")
        scene_mapping = {i: f'scene_{i+1}' for i in range(len(scene_detect))}
        scene_detect.index = [scene_mapping[i] for i in scene_detect.index]

        scene_idx = natsorted(clustered_face.keys())
        df = pd.DataFrame(index=pd.Index(scene_idx), columns=[str(i) for i in range(6)])

        for idx, faces in clustered_face.items():
            if faces:
                for face in faces:
                    cls = face['cluster_id']
                    ch = matched_face.get(str(cls))
                    if ch:
                        df.loc[idx, ch] = 1

        # Rename columns for better clarity
        column_mapping = {
            '0': 'chandler', '1': 'joey', '2': 'monica',
            '3': 'phoebe', '4': 'rachel', '5': 'ross'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Reindex scene_detect with all possible scenes
        all_scenes = [f'scene_{i+1}' for i in range(len(scene_detect))]
        scene_detect = scene_detect.reindex(all_scenes)

        df = df.reindex(all_scenes, fill_value=0)
        merged_df = pd.concat([scene_detect, df], axis=1)

        # Fill NaN values with 0
        merged_df = merged_df.fillna(0)

        # Change to float
        merged_df = merged_df.astype(float)
        
        output_file = generate_file_path(save_dir, f"{episode_id}_event.csv")
        merged_df.to_csv(output_file, index=True)
        print(f"Processed episode {episode_id} successfully.")

    except Exception as e:
        print(f"Error processing episode {episode_id}: {e}")

def main():
    load_dotenv()

    # Load environment variables
    scratch_dir = os.getenv("SCRATCH_DIR")

    if not scratch_dir:
        raise EnvironmentError("SCRATCH_DIR environment variable is not set.")

    save_dir = generate_file_path(scratch_dir, "output", "07_face_event")
    os.makedirs(save_dir, exist_ok=True)

    # Read episode IDs from the file
    episode_file = generate_file_path(scratch_dir, "data", "episode_id.txt")
    try:
        with open(episode_file, "r") as f:
            episode_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Episode ID file not found: {episode_file}")

    if not episode_ids:
        print("No episode IDs found. Exiting.")
        return

    # Process each episode
    for episode_id in episode_ids:
        process_episode(episode_id, scratch_dir, save_dir)
if __name__ == "__main__":
    main()