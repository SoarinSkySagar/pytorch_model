import torch
from imports.data import train_dataloader, test_dataloader
from imports.device import device
from imports.network import model
from imports.train_test import train, test, loss_fn, optimizer

print(f"Using {device} device")

model.load_state_dict(torch.load("model/model.pth"))

epochs = 1000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    torch.save(model.state_dict(), "model/model.pth")

print("Done!")
print("Saved PyTorch Model State to model/model.pth")