import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
print("Loading data...")
data = pd.read_csv('data.csv', delimiter=";")

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_data, val_data = train_test_split(
    data, 
    test_size=0.1,       # 20% for validation
    random_state=42,     # For reproducibility
    shuffle=True
)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# set up data loading for the training and validation set
print("\nSetting up data loaders...")
train_dataset = ChallengeDataset(train_data, mode='train')
val_dataset = ChallengeDataset(val_data, mode='val')

train_loader = t.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

val_loader = t.utils.data.DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4
)

# create an instance of our ResNet model
print("\nCreating model...")
resnet = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.BCELoss()
# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(resnet.parameters(), lr=0.001)

# Check if CUDA is available
cuda_available = t.cuda.is_available()
print(f"Using device: {'CUDA (GPU)' if cuda_available else 'CPU'}")

# create an object of type Trainer and set its early stopping criterion
print("\nInitializing trainer...")
# Create checkpoints directory
os.makedirs('checkpoints', exist_ok=True)

trainer = Trainer(
    model=resnet,
    crit=criterion,
    optim=optimizer,
    train_dl=train_loader,
    val_test_dl=val_loader,
    cuda=cuda_available,
    early_stopping_patience=5  # Stop if no improvement for 5 epochs
)

# go, go, go... call fit on trainer
print("\nStarting training...\n")
res = trainer.fit(epochs=2)


# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')

# Save the final model
print("\nSaving model...")
trainer.save_onnx('model.onnx')
print("Model saved to 'model.onnx'")

print("\n✅ Training complete!")