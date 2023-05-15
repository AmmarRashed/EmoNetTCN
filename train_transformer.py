import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmbeddingDataset
from transformer import ImageEmbeddingRegressor

# set random seed for Python's built-in random module
random.seed(0)

# set random seed for NumPy
np.random.seed(0)

# set random seed for PyTorch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# initialize the logger
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')
parser = ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the model (default: 50)')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (default: 5)')
parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
parser.add_argument('--root_dir', type=str, default='./data', help='Root directory of the dataset (default: ./data)')
args = parser.parse_args()

num_epochs = args.num_epochs

train_dataset = EmbeddingDataset(os.path.join(args.root_dir, "Train"))
val_dataset = EmbeddingDataset(os.path.join(args.root_dir, "Validation"))
test_dataset = EmbeddingDataset(os.path.join(args.root_dir, "Test"))

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

model = ImageEmbeddingRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

best_val_loss = float('inf')
epochs_since_last_improvement = 0
patience = args.patience
best_model_state_dict = model.state_dict()
for epoch in tqdm(range(num_epochs)):
    # Training loop
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader, leave=False):
        optimizer.zero_grad()
        output = model(batch["x"])
        loss = criterion(output, batch["y"].float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            output = model(batch["x"])
            loss = criterion(output, batch["y"].float())
            val_loss += loss.item()
    val_loss = val_loss / len(val_dataloader)
    train_loss = train_loss / len(train_dataloader)
    # Print epoch statistics
    print(f"Epoch {epoch + 1} Train Loss: {train_loss} Val Loss: {val_loss}")
    logging.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_since_last_improvement = 0
        best_model_state_dict = model.state_dict()
    if epochs_since_last_improvement == patience:
        print(f"No improvements after {patience} epochs. Training stopped.")
        break

torch.save(best_model_state_dict, 'best_model_params.pt')

# Load the best model parameters and test the model
model.load_state_dict(best_model_state_dict)
model.eval()
test_loss = 0
with torch.no_grad():
    for batch_x, batch_y in test_dataloader:
        output = model(batch_x)
        loss = criterion(output, batch_y.float())
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_dataloader)}")
