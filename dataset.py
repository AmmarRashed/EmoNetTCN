import os

import pandas as pd
import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch.utils.data import Dataset

from face_detection import detect_faces, FastMTCNN


def get_path(clip_id, root, ttv):
    participant_id = clip_id[:6]
    path = os.path.join(root, ttv.title(), participant_id, clip_id.split(".")[0], clip_id)
    try:
        assert os.path.isfile(path)
    except AssertionError:
        return None
    return path


class VideoDataset(Dataset):
    def __init__(self, root, csv, ttv, face_detector=None, embedder=None):
        if face_detector is None:
            face_detector = FastMTCNN()

        if embedder is None:
            embedder = InceptionResnetV1(pretrained='vggface2').eval()
        
        self.face_detector = face_detector
        self.embedder = embedder
        
        self.root = root
        self.csv = csv
        self.ttv = ttv  # train test or validation
        self.paths_df = pd.read_csv(csv)
        self.paths_df = self.paths_df.assign(
            path=self.paths_df.ClipID.apply(get_path, root=root, ttv=ttv)
        ).dropna()
        self.paths_df.columns = [c.strip() for c in self.paths_df.columns]  # some columns have extra spaces

    def __len__(self):
        return len(self.paths_df)

    def __getitem__(self, idx):
        clip = self.paths_df.iloc[idx]
        clip_id, extension = os.path.splitext(clip.ClipID)
        try:
            faces = detect_faces(self.face_detector, clip.path, max_frames=300)
            embedding = self.embedder(faces).detach()
        except Exception as e:
            warnings.warn(f"Error processing clip {clip['clip_id']}. Skipping.\n{type(e)}: {e}")
            return
        return dict(
            clip_id=clip_id,
            extension=extension,
            ttv=self.ttv,
            path=clip.path,
            embedding=embedding,
            boredom=clip.Boredom,
            engagement=clip.Engagement,
            confusion=clip.Confusion,
            frustration=clip.Frustration)
