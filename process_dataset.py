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


def process_batch(batch):
    embeddings = []
    for clip_id, clip_path in zip(batch["clip_id"], batch["path"]):
        try:
            faces = detect_faces(mtcnn, clip_path, max_frames=300)
            embedding = resnet(faces).detach()
        except Exception as e:
            warnings.warn(f"Error processing clip {clip_id}. Skipping.\n{type(e)}: {e}")
            continue
        embeddings.append(embedding)
    batch["embeddings"] = torch.stack(embeddings, dim=0)
    return batch


for ttv in ["Train", "Validation", "Test"]:
    print(f"Processing {ttv} dataset")
    dataset = VideoDataset(root="../DAiSEE/DataSet/", csv=f"../DAiSEE/Labels/{ttv}Labels.csv", ttv=ttv)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for i, batch in tqdm(enumerate(dataloader)):
        path = os.path.join("vectors", ttv, f"batch_{i}")
        if os.path.isfile(path + ".pt"):
            print(f"skipped batch {i}")
            continue
        batch = process_batch(batch)
        torch.save(batch, path + ".pt", pickle_module=pickle, pickle_protocol=4)
        # del batch["faces"]
        # # faces = batch.pop("faces")
        # # pickle.dump(batch, path, 'wb'))
        # torch.save(batch, path+".pt", pickle_module=pickle, pickle_protocol=4)
        # # torch.save(faces, path+"_faces.pt")
        # torch.cuda.empty_cache()
