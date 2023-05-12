import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video


def get_path(clip_id, root, ttv):
    participant_id = clip_id[:6]
    path = os.path.join(root, ttv.title(), participant_id, clip_id.split(".")[0], clip_id)
    try:
        assert os.path.isfile(path)
    except AssertionError:
        return None
    return path


class VideoDataset(Dataset):
    def __init__(self, root, csv, ttv, transform=None):
        self.root = root
        self.csv = csv
        self.ttv = ttv  # train test or validation
        self.paths_df = pd.read_csv(csv)
        self.paths_df = self.paths_df.assign(
            path=self.paths_df.ClipID.apply(get_path, root=root, ttv=ttv)
        ).dropna()
        self.paths_df.columns = [c.strip() for c in self.paths_df.columns]  # some columns have extra spaces
        self.transform = transform
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.paths_df)

    def __getitem__(self, idx):
        clip = self.paths_df.iloc[idx]
        video, audio, info = read_video(clip.path)
        if self.transform:
            video = self.transform(video)
        video = video.to(self.device)
        return dict(video=video,
                    boredom=clip.Boredom,
                    engagement=clip.Engagement,
                    confusion=clip.Confusion,
                    frustration=clip.Frustration)
