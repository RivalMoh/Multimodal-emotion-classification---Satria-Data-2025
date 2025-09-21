# datasets.py
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
from transformers import AutoTokenizer

def build_label_encoder(labels):
    unique_labels = sorted(set(labels))
    label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx2label = {i: lbl for lbl, i in label2idx.items()}
    return label2idx, idx2label

class VideoDataset(Dataset):
    def __init__(self, manifest_csv, split="train", fold=0, transform=None, label2idx=None):
        df = pd.read_csv(manifest_csv)
        self.df = df[(df["split"]==split) & (df["fold"]==fold)].reset_index(drop=True)
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = torch.load(row["frames_path"])
        if self.transform: frames = self.transform(frames)
        label = self.label2idx[row["label"]]
        return frames, label, {"video_id":row["video_id"],"window_idx":row["window_idx"]}

class AudioDataset(Dataset):
    def __init__(self, manifest_csv, split="train", fold=0, sr=16000, label2idx=None):
        df = pd.read_csv(manifest_csv)
        self.df = df[(df["split"]==split) & (df["fold"]==fold)].reset_index(drop=True)
        self.sr = sr
        self.label2idx = label2idx

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav, _ = librosa.load(row["audio_path"], sr=self.sr)
        wav = torch.tensor(wav, dtype=torch.float32)
        label = self.label2idx[row["label"]]
        return wav, label, {"video_id":row["video_id"],"window_idx":row["window_idx"]}

class TextDataset(Dataset):
    def __init__(self, manifest_csv, split="train", fold=0, model_name="bert-base-uncased", max_len=64, label2idx=None):
        df = pd.read_csv(manifest_csv)
        self.df = df[(df["split"]==split) & (df["fold"]==fold)].reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.label2idx = label2idx

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text_snippet"])
        tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        label = self.label2idx[row["label"]]
        return {k:v.squeeze(0) for k,v in tokens.items()}, label, {"video_id":row["video_id"],"window_idx":row["window_idx"]}