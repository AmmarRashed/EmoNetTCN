import os
import pickle
import warnings

import torch
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoDataset
from face_detection import FastMTCNN, detect_faces

mtcnn = FastMTCNN(image_size=160, margin=0, min_face_size=20,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)

resnet = InceptionResnetV1(pretrained='vggface2').eval()

output = "vectors"


def process_batch(batch):
    for clip in batch:
        try:
            faces = detect_faces(mtcnn, clip["path"], max_frames=300)
            embedding = resnet(faces).detach()
        except Exception as e:
            warnings.warn(f"Error processing clip {clip['clip_id']}. Skipping.\n{type(e)}: {e}")
            continue
        clip['embedding'] = embedding
        torch.save(clip, os.path.join(output, clip['ttv'], clip['clip_id']+".pt"),
                   pickle_module=pickle, pickle_protocol=4)


for ttv in ["Train", "Validation", "Test"]:
    print(f"Processing {ttv} dataset")
    dataset = VideoDataset(root="../DAiSEE/DataSet/", csv=f"../DAiSEE/Labels/{ttv}Labels.csv", ttv=ttv)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=process_batch)

    for _ in tqdm(dataloader):
        pass
