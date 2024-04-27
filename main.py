import torch
from imports.data import train_dataloader, test_dataloader
from imports.device import device
from imports.network import model
from imports.train_test import train, test, loss_fn, optimizer

print(f"Using {device} device")

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

print(model)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model/model.pth")
print("Saved PyTorch Model State to model.pth")