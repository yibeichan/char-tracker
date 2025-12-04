import glob
import os
import sys
from dotenv import load_dotenv
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import argparse

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

def get_face_embedding(image_path, model, device, model_name='vggface2', embedding_dim=512):
    """
    Compute the face embedding of an image using a face recognition model.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    model : Model
        The model to use for computing face embeddings.
    device : torch.device
        The device to run the model on.
    model_name : str
        Name of the model ('vggface2' or 'buffalo_l').
    embedding_dim : int
        Dimension of the embedding (512 for both models).

    Returns
    -------
    embedding : np.ndarray
        The face embedding of the image, or a vector of NaNs if an error occurred.
    """
    try:
        face_tensor = load_image(image_path, model_name)
        if model_name == 'buffalo_l':
            # Convert normalized tensor back to uint8 image for insightface
            face_img = ((face_tensor * 127.5) + 127.5).astype(np.uint8)
            faces = model.get(face_img)
            if len(faces) > 0:
                embedding = faces[0].normed_embedding
            else:
                print(f"Warning: No face detected in {image_path}")
                embedding = np.zeros(embedding_dim)
        else:
            face_tensor = face_tensor.to(device)
            with torch.no_grad():
                embedding = model(face_tensor).cpu().numpy().flatten()
        return embedding
    except Exception as e:
        print(f"Error processing image: {image_path}: {e}")
        return np.full((embedding_dim,), np.nan)

def load_image(image_path, model_name='vggface2'):
    """Load an image from disk and convert it to a tensor."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    return preprocess_face(image, model_name)

def preprocess_face(face_image, model_name='vggface2'):
    """Preprocess the face image for embedding extraction (resize, normalize, etc.)."""
    if model_name == 'buffalo_l':
        # insightface buffalo_l: 112x112, different normalization
        face_image = cv2.resize(face_image, (112, 112))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_tensor = (face_image.astype(np.float32) - 127.5) / 127.5
    else:
        # InceptionResnetV1: 160x160
        face_image = cv2.resize(face_image, (160, 160))
        face_tensor = torch.tensor(face_image).permute(2, 0, 1).float().unsqueeze(0)
        face_tensor = (face_tensor - 127.5) / 128.0
    return face_tensor

def main(input_dir, season_id, save_dir, start_episode=None, end_episode=None, model_name='vggface2'):
    """
    Compute face embeddings for all characters in a given range of episodes.

    Parameters
    ----------
    input_dir : str
        The directory containing the face images.
    season_id : int
        The season ID (1-6).
    save_dir : str
        The directory to save the face embeddings.
    start_episode : int, optional
        The first episode to process. If not provided, all episodes in the input directory are processed.
    end_episode : int, optional
        The last episode to process. If not provided, all episodes in the input directory are processed.
    model_name : str, optional
        The model to use for embeddings ('vggface2' or 'buffalo_l').

    Notes
    -----
    The script assumes that the face images are stored in the input directory with the following naming convention:
        friends_<season_id>e<episode_number>*char_<character_id>.jpg
    For each episode window (of size 2), the script collects images for all characters, checks the minimum image count, and
    processes the embeddings for each character.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'buffalo_l':
        import insightface
        from insightface.app import FaceAnalysis
        model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device.type == 'cuda' else 'CPUExecutionProvider'])
        model.prepare(ctx_id=0 if device.type == 'cuda' else -1, det_size=(640, 640))
        embedding_dim = 512  # buffalo_l uses 512-dim embeddings
    else:
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        embedding_dim = 512
    
    if start_episode is None or end_episode is None:
        all_episodes = sorted({extract_episode_number(f, season_id) for f in glob.glob(os.path.join(input_dir, f'friends_{season_id}e*.jpg')) if extract_episode_number(f, season_id) is not None})
        print("Found episodes:", all_episodes)  # Debugging output

        if not all_episodes:
            print(f"No episodes found in directory {input_dir} with pattern friends_{season_id}e*.jpg")
            sys.exit(1)
        
        start_episode = min(all_episodes)
        end_episode = max(all_episodes) + 1  # Adding 1 to include the last episode
    
    for episode in range(start_episode, end_episode):
        char_images = {char_id: [] for char_id in range(6)}
        
        # Determine the end of the episode window
        if episode + 2 <= end_episode:
            episode_window_end = episode + 2
        else:
            episode_window_end = end_episode
        # Collect images for the episode window
        for e in range(episode, episode_window_end):
            for char_id in range(6):
                pattern = os.path.join(input_dir, f'friends_{season_id}e{e:02d}*char_{char_id}.jpg')
                files = glob.glob(pattern)
                char_images[char_id].extend(files)
                print(f"Found {len(files)} files for char_{char_id} in episode {e}")
        
        # Check minimum image counts after collecting all images
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
                embedding = get_face_embedding(img, model, device, model_name, embedding_dim)
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
    parser.add_argument('--model-name', type=str, default='vggface2',
                       choices=['vggface2', 'buffalo_l'],
                       help='Embedding model to use (default: vggface2)')
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

    main(input_dir, season_id, save_dir, model_name=args.model_name)
