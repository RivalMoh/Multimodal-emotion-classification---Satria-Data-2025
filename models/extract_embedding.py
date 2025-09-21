import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

def video_collate_fn(batch):
    """Custom collate function to handle variable frame counts in videos"""
    videos = []
    labels = []
    metadata = []
    
    for item in batch:
        video, label, meta = item
        videos.append(video)
        labels.append(label)
        metadata.append(meta)
    
    # Find the maximum number of frames
    max_frames = max(v.shape[0] for v in videos)
    
    # Pad videos to have the same number of frames
    padded_videos = []
    for video in videos:
        num_frames, channels, height, width = video.shape
        if num_frames < max_frames:
            # Pad by repeating the last frame
            padding_needed = max_frames - num_frames
            last_frame = video[-1:].repeat(padding_needed, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
        padded_videos.append(video)
    
    # Stack videos
    videos = torch.stack(padded_videos)
    
    # Validate labels are in correct range for 8 classes (0-7)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    max_label = labels_tensor.max().item() if len(labels) > 0 else -1
    min_label = labels_tensor.min().item() if len(labels) > 0 else -1
    
    if max_label >= 8 or min_label < 0:
        print(f"WARNING: Invalid labels detected! Expected 0-7, got {min_label}-{max_label}")
        print(f"Labels in batch: {labels}")
        # Clip labels to valid range
        labels_tensor = torch.clamp(labels_tensor, 0, 7)
    
    return videos, labels_tensor, metadata

def audio_collate_fn(batch):
    """Custom collate function for audio data"""
    audios = []
    labels = []
    metadata = []
    
    for item in batch:
        audio, label, meta = item
        audios.append(audio)
        labels.append(label)
        metadata.append(meta)
    
    # Handle variable length audio sequences
    max_length = max(a.shape[-1] for a in audios)
    
    padded_audios = []
    for audio in audios:
        if audio.shape[-1] < max_length:
            # Pad with zeros
            padding_needed = max_length - audio.shape[-1]
            if len(audio.shape) == 1:
                audio = torch.cat([audio, torch.zeros(padding_needed)], dim=0)
            else:
                audio = torch.cat([audio, torch.zeros(*audio.shape[:-1], padding_needed)], dim=-1)
        padded_audios.append(audio)
    
    audios = torch.stack(padded_audios)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    labels_tensor = torch.clamp(labels_tensor, 0, 7)
    
    return audios, labels_tensor, metadata

def text_collate_fn(batch):
    """Custom collate function for text data"""
    texts = []
    labels = []
    metadata = []
    
    for item in batch:
        text, label, meta = item
        texts.append(text)
        labels.append(label)
        metadata.append(meta)
    
    # Stack text tensors (assuming they're already tokenized and padded)
    texts = torch.stack(texts)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    labels_tensor = torch.clamp(labels_tensor, 0, 7)
    
    return texts, labels_tensor, metadata

def extract_embeddings(model, dataset, out_dir, device, modality='video', batch_size=4):
    """
    Extract embeddings from specialist models
    
    Args:
        model: Trained specialist model (video, audio, or text)
        dataset: Dataset containing the data
        out_dir: Output directory to save embeddings
        device: Device to run inference on
        modality: Type of modality ('video', 'audio', 'text')
        batch_size: Batch size for inference
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.to(device).eval()
    
    # Choose appropriate collate function based on modality
    collate_fn = {
        'video': video_collate_fn,
        'audio': audio_collate_fn,
        'text': text_collate_fn
    }.get(modality, video_collate_fn)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=collate_fn
    )
    
    rows = []
    
    print(f"Extracting {modality} embeddings...")
    for batch_idx, (x, y, meta) in enumerate(loader):
        x = x.to(device)
        
        with torch.no_grad():
            # Extract embeddings from the model
            if hasattr(model, 'extract_features'):
                # If model has specific feature extraction method
                emb = model.extract_features(x)
            else:
                # Forward pass through the model
                emb = model(x)
            
            # Flatten embeddings for storage
            emb = emb.view(emb.size(0), -1).cpu().numpy()
        
        # Process each sample in the batch
        for i in range(len(y)):
            # Extract metadata for current sample
            video_id = meta[i].get('video_id', f'sample_{batch_idx * batch_size + i}')
            window_index = meta[i].get('window_index', i)
            label_str = meta[i].get('label_str', str(y[i].item()))
            actual_label = y[i].item()
            
            # Create embedding filename
            embedding_filename = f"{video_id}_{window_index}_{modality}.npy"
            embedding_path = os.path.join(out_dir, embedding_filename)
            
            # Save embedding to file
            np.save(embedding_path, emb[i])
            
            # Store metadata
            rows.append({
                "video_id": video_id,
                "window_index": window_index,
                "label": actual_label,
                "label_str": label_str,
                "modality": modality,
                "embedding_shape": emb[i].shape,
                "embedding_path": embedding_path
            })
        
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx * batch_size} samples...")
    
    print(f"Extracted embeddings for {len(rows)} samples")
    return pd.DataFrame(rows)

def extract_multimodal_embeddings(video_model, audio_model, text_model, 
                                video_dataset, audio_dataset, text_dataset,
                                out_dir, device, batch_size=4):
    """
    Extract embeddings from all three specialist models
    
    Args:
        video_model, audio_model, text_model: Trained specialist models
        video_dataset, audio_dataset, text_dataset: Respective datasets
        out_dir: Base output directory
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns:
        Dictionary containing DataFrames for each modality
    """
    results = {}
    
    # Extract video embeddings
    if video_model is not None and video_dataset is not None:
        video_out_dir = os.path.join(out_dir, 'video_embeddings')
        results['video'] = extract_embeddings(
            video_model, video_dataset, video_out_dir, device, 'video', batch_size
        )
    
    # Extract audio embeddings
    if audio_model is not None and audio_dataset is not None:
        audio_out_dir = os.path.join(out_dir, 'audio_embeddings')
        results['audio'] = extract_embeddings(
            audio_model, audio_dataset, audio_out_dir, device, 'audio', batch_size
        )
    
    # Extract text embeddings
    if text_model is not None and text_dataset is not None:
        text_out_dir = os.path.join(out_dir, 'text_embeddings')
        results['text'] = extract_embeddings(
            text_model, text_dataset, text_out_dir, device, 'text', batch_size
        )
    
    # Save combined metadata
    if results:
        combined_df = pd.concat(results.values(), ignore_index=True)
        metadata_path = os.path.join(out_dir, 'embeddings_metadata.csv')
        combined_df.to_csv(metadata_path, index=False)
        print(f"Saved combined metadata to {metadata_path}")
    
    return results

def load_embeddings_for_fusion(metadata_df, video_id, window_index):
    """
    Load embeddings for fusion model training
    
    Args:
        metadata_df: DataFrame containing embedding metadata
        video_id: Video ID to load
        window_index: Window index to load
    
    Returns:
        Dictionary containing embeddings for each modality
    """
    embeddings = {}
    
    # Filter for specific video and window
    mask = (metadata_df['video_id'] == video_id) & (metadata_df['window_index'] == window_index)
    relevant_rows = metadata_df[mask]
    
    for _, row in relevant_rows.iterrows():
        modality = row['modality']
        embedding_path = row['embedding_path']
        
        if os.path.exists(embedding_path):
            embeddings[modality] = np.load(embedding_path)
        else:
            print(f"Warning: Embedding file not found: {embedding_path}")
    
    return embeddings