"""
Character Embedding Extraction Script

Extracts reference face embeddings for the 6 main Friends characters
using the InsightFace buffalo_l model.

The script processes episode images in sliding windows (2 episodes per window)
to ensure sufficient training data across episodes.
"""

import glob
import os
import sys
from dotenv import load_dotenv
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# Import insightface for buffalo_l model
import insightface
from insightface.app import FaceAnalysis


def extract_episode_number(file_path, season_id):
    """
    Extract the episode number from a file path.

    Parameters
    ----------
    file_path : str
        The file path to extract the episode number from.
    season_id : str
        The season ID to extract the episode number from.

    Returns
    -------
    int or None
        The episode number as an integer if successfully extracted, otherwise None.
    """
    try:
        episode_part = file_path.split(f'{season_id}e')[1][:2]
        return int(episode_part)
    except (IndexError, ValueError) as e:
        print(f"Error extracting episode number from {file_path}: {e}")
        return None


def get_face_embedding(image_path, model, embedding_dim=512):
    """
    Compute the face embedding of an image using buffalo_l model.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    model : FaceAnalysis
        The InsightFace FaceAnalysis model.
    embedding_dim : int
        Dimension of the embedding (512 for buffalo_l).

    Returns
    -------
    embedding : np.ndarray
        The face embedding of the image, or a vector of NaNs if an error occurred.
    """
    try:
        face_tensor = load_image(image_path)
        # Pass directly to recognition model (bypasses face detection)
        faces = model.get(face_tensor)
        if len(faces) > 0:
            embedding = faces[0].normed_embedding
        else:
            print(f"Warning: No face detected in {image_path}")
            embedding = np.full((embedding_dim,), np.nan)
        return embedding
    except Exception as e:
        print(f"Error processing image: {image_path}: {e}")
        return np.full((embedding_dim,), np.nan)


def load_image(image_path):
    """
    Load an image from disk and preprocess for buffalo_l.

    buffalo_l expects 112x112 BGR images (uint8, 0-255).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    return cv2.resize(image, (112, 112))


def main(input_dir, season_id, save_dir, start_episode=None, end_episode=None):
    """
    Compute face embeddings for all characters in a given range of episodes.

    Parameters
    ----------
    input_dir : str
        The directory containing the face images.
    season_id : str
        The season ID (e.g., '1', '2', etc.).
    save_dir : str
        The directory to save the face embeddings.
    start_episode : int, optional
        The first episode to process. If not provided, all episodes in the input directory are processed.
    end_episode : int, optional
        The last episode to process. If not provided, all episodes in the input directory are processed.

    Notes
    -----
    The script assumes that the face images are stored in the input directory with the following naming convention:
        friends_<season_id>e<episode_number>*char_<character_id>.jpg
    For each episode window (of size 2), the script collects images for all characters, checks the minimum image count, and
    processes the embeddings for each character.
    """
    # Initialize InsightFace buffalo_l model
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("Using GPU (CUDA) for inference")
        else:
            providers = ['CPUExecutionProvider']
            print("Using CPU for inference (CUDA not available)")
    except ImportError:
        providers = ['CPUExecutionProvider']
        print("Using CPU for inference (onnxruntime not installed)")

    model = FaceAnalysis(name='buffalo_l', providers=providers)
    model.prepare(ctx_id=0 if 'CUDAExecutionProvider' in providers else -1, det_size=(640, 640))
    embedding_dim = 512

    if start_episode is None or end_episode is None:
        all_episodes = sorted({
            extract_episode_number(f, season_id)
            for f in glob.glob(os.path.join(input_dir, f'friends_{season_id}e*.jpg'))
            if extract_episode_number(f, season_id) is not None
        })
        print("Found episodes:", all_episodes)

        if not all_episodes:
            print(f"No episodes found in directory {input_dir} with pattern friends_{season_id}e*.jpg")
            sys.exit(1)

        start_episode = min(all_episodes)
        end_episode = max(all_episodes) + 1

    print(f"Processing episodes {start_episode} to {end_episode - 1}")

    for episode in range(start_episode, end_episode):
        char_images = {char_id: [] for char_id in range(6)}

        # Determine the end of the episode window
        episode_window_end = min(episode + 2, end_episode)

        # Collect images for the episode window
        for e in range(episode, episode_window_end):
            for char_id in range(6):
                pattern = os.path.join(input_dir, f'friends_{season_id}e{e:02d}*char_{char_id}.jpg')
                files = glob.glob(pattern)
                char_images[char_id].extend(files)
                print(f"Found {len(files)} files for char_{char_id} in episode {e}")

        # Check minimum image counts
        image_counts = [len(files) for files in char_images.values()]
        min_images = min(image_counts)
        print(f'Minimum images across {episode_window_end - episode} episodes for any character: {min_images}')

        if min_images == 0:
            print("Skipping processing for this episode window due to 0 images for one or more characters.")
            continue

        # Process embeddings
        for char_id, files in char_images.items():
            if len(files) == 0:
                continue
            selected_images = np.random.choice(files, min_images, replace=False)
            embeddings = []

            for img in tqdm(selected_images, desc=f"Processing char_{char_id} embeddings"):
                embedding = get_face_embedding(img, model, embedding_dim)
                embeddings.append(embedding)

            embeddings = np.array(embeddings)
            print(f"Embeddings shape: {embeddings.shape}")
            filename = os.path.join(save_dir, f'{season_id}_e{episode:02d}-e{episode_window_end-1:02d}_char_{char_id}_embeddings.npy')
            np.save(filename, embeddings)
            print(f"Saved {filename}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Extract embeddings for all characters in a season.")
    parser.add_argument("season_id", help="Season ID for processing")
    args = parser.parse_args()

    season_id = args.season_id

    required_env_vars = ["SCRATCH_DIR"]

    # Check that all required environment variables are set
    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    if missing_vars:
        print(f"Error: Missing required environment variable(s): {', '.join(missing_vars)}")
        sys.exit(1)

    scratch_dir = os.getenv("SCRATCH_DIR")
    output_dir = os.path.join(scratch_dir, "output")
    input_dir = os.path.join(output_dir, "char_face")
    save_dir = os.path.join(output_dir, "char_ref_embs")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    main(input_dir, season_id, save_dir)
