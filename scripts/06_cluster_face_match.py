import os
import glob
import json
import numpy as np
from dotenv import load_dotenv
import argparse
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def extract_season_episode(file_name):
    # Regular expression to capture 'sXX' and 'eXX' patterns
    match = re.search(r's(\d{2})e(\d{2})', file_name)
    
    if match:
        s_id = f"s{match.group(1)}" 
        e_id = int(match.group(2))
        return s_id, e_id
    else:
        raise ValueError(f"Filename '{file_name}' does not contain a valid season and episode format")

def get_episode_ranges(episode, total_episodes):
    if episode == total_episodes:
        # Special case for the last episode, we can't look forward
        return f"e{total_episodes-1:02d}-e{total_episodes:02d}"  # List with one range
    else:
        # General case for episodes in the middle
        return f"e{episode:02d}-e{episode+1:02d}"  # Current episode and the next one


def reorganize_by_cluster(clustered_faces):
    clusters = {}
    for scene_id, faces in clustered_faces.items():
        for face_data in faces:
            cluster_id = face_data['cluster_id']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(face_data)
    return clusters

def load_reference_embeddings(season, episode_range, ref_emb_dir):
    embeddings_files = []
    embeddings_path = os.path.join(ref_emb_dir, f"{season}_{episode_range}_char_*_embeddings.npy")
    found_files = glob.glob(embeddings_path)
    if not found_files:
        print(f"Warning: No embedding files found for episode {episode_range} in {ref_emb_dir}.")
    embeddings_files.extend(found_files)  # Extend to flatten the list of files

    embeddings = {}
    for file in embeddings_files:
        try:
            char_id = os.path.basename(file).split('_char_')[1].split('_')[0]
        except IndexError:
            print(f"Error parsing character ID from filename: {file}")
            continue  # Skip the file if the naming is incorrect
        embeddings[char_id] = np.load(file)
    return embeddings

def find_nearest_exemplar(cluster_embeddings):
    """Find the exemplar (medoid) that is closest to all other points in the cluster."""
    similarities = cosine_similarity(cluster_embeddings)
    total_similarities = np.sum(similarities, axis=1)
    # The exemplar is the embedding with the highest total similarity to others in the cluster
    exemplar_idx = np.argmax(total_similarities)
    return cluster_embeddings[exemplar_idx]

def character_similarity_distribution(character_embeddings):
    """Calculate the mean and standard deviation of internal similarity for a character."""
    char_matrix = np.array([np.array(embedding).flatten() for embedding in character_embeddings])
    similarities = cosine_similarity(char_matrix)
    internal_mean = np.mean(similarities)
    internal_std = np.std(similarities)
    return internal_mean, internal_std

def dynamic_threshold(internal_mean, internal_std, scale_factor=0.5):
    """Set a dynamic threshold based on the character's internal similarity distribution."""
    return internal_mean - scale_factor * internal_std

def top_percent_average(similarities, percent=10, top_k_fallback=3):
    """Compute the average of the top N% of similarities, with a fallback to top-k if necessary."""
    num_top = max(1, int(len(similarities) * percent / 100))
    if num_top < top_k_fallback:  # Fallback to top-k if not enough embeddings
        num_top = top_k_fallback
    top_similarities = np.sort(similarities)[-num_top:]
    return np.mean(top_similarities)

def identify_character(cluster_embeddings, character_dict, scale_factor=0.5, top_k_fallback=3):
    """Identify the best matching character for a cluster based on exemplar, dynamic threshold, and top similarities."""
    # Step 1: Find the exemplar of the cluster
    cluster_matrix = np.array([np.array(embedding).flatten() for embedding in cluster_embeddings])
    cluster_exemplar = find_nearest_exemplar(cluster_matrix)

    best_similarity = -1
    best_char_id = "unknown"

    # Step 2: Compare the exemplar to each character's embeddings
    for char_id, char_embeddings in character_dict.items():
        # Compute internal similarity for this character
        internal_mean, internal_std = character_similarity_distribution(char_embeddings)
        
        # Compute dynamic threshold for this character
        threshold = dynamic_threshold(internal_mean, internal_std, scale_factor)

        # Compare the cluster exemplar with the character's embeddings
        char_matrix = np.array([np.array(embedding).flatten() for embedding in char_embeddings])
        similarities = cosine_similarity(cluster_exemplar.reshape(1, -1), char_matrix).flatten()

        # Use top 10% averaging, with a fallback to top-k if necessary
        avg_similarity = top_percent_average(similarities, percent=10, top_k_fallback=top_k_fallback)

        if avg_similarity > threshold and avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_char_id = char_id

    return best_char_id
    
def main(episode_id, clustered_faces, ref_emb_dir, output_dir):
    clusters = reorganize_by_cluster(clustered_faces)
    
    season, episode = extract_season_episode(episode_id)
    if season == "s03":
        total_episode = 25
    elif season == "s04" or season == "s05":
        total_episode = 23
    else:
        total_episode = 24
    episode_range = get_episode_ranges(episode, total_episode)

    reference_embeddings = load_reference_embeddings(season, episode_range, ref_emb_dir)

    cluster_char_assignments = {}
    for cluster_id, faces in clusters.items():
        cluster_embeddings = [embedding for face in faces for embedding in face['embeddings']]
        best_character = identify_character(cluster_embeddings, reference_embeddings)
        print(f"Cluster {cluster_id}: Best Character = {best_character}, with {len(faces)} faces")
        cluster_char_assignments[cluster_id] = best_character

    with open(os.path.join(output_dir, f"{episode_id}_cluster-face_matching.json"), 'w') as f:
        json.dump(cluster_char_assignments, f)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Match characters in an episode.")
    parser.add_argument("episode_id", help="Episode ID for processing")
    args = parser.parse_args()
    episode_id = args.episode_id

    scratch_dir = os.getenv("SCRATCH_DIR")

    cluster_file = os.path.join(scratch_dir, "output", "04_face_clustering", f"{episode_id}_matched_faces_with_clusters.json")
    ref_emb_dir = os.path.join(scratch_dir, "output", "05_char_ref_embs")
    output_dir = os.path.join(scratch_dir, "output", "06_cluster_face_matching")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(cluster_file, 'r') as f:
        clustered_data = json.load(f)

    main(episode_id, clustered_data, ref_emb_dir, output_dir)