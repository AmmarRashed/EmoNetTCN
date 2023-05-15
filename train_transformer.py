import os
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmbeddingDataset
from transformer import ImageEmbeddingRegressor

parser = ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the model (default: 50)')
parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
parser.add_argument('--root_dir', type=str, default='./data', help='Root directory of the dataset (default: ./data)')
args = parser.parse_args()

num_epochs = args.num_epochs
best_val_loss = float('inf')

train_dataset = EmbeddingDataset(os.path.join(args.root_dir, "Train"))
val_dataset = EmbeddingDataset(os.path.join(args.root_dir, "Validation"))
test_dataset = EmbeddingDataset(os.path.join(args.root_dir, "Test"))

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

model = ImageEmbeddingRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

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

    # Print epoch statistics
    print(
        f"Epoch {epoch + 1} Train Loss: {train_loss / len(train_dataloader)} Val Loss: {val_loss / len(val_dataloader)}")

    # Save model's best parameters every 10 epochs
    if (epoch + 1) % 10 == 0 and val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_params.pt')

# Load the best model parameters and test the model
model.load_state_dict(torch.load('best_model_params.pt'))
model.eval()
test_loss = 0
with torch.no_grad():
    for batch_x, batch_y in test_dataloader:
        output = model(batch_x)
        loss = criterion(output, batch_y.float())
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_dataloader)}")
