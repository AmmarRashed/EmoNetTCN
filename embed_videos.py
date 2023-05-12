from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader

from dataset import VideoDataset

dataset = VideoDataset(root="../DAiSEE/DataSet/", csv="../DAiSEE/Labels/TrainLabels.csv", ttv="Train")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

