import torch, time, os
from tqdm import tqdm
from dataset import HySpecNet11k
from torch.utils.data import DataLoader
import pandas as pd

def train(model, train_loader, valid_loader, num_epochs, optimizer, scheduler, loss_fun, device, save_pretrained):
    model.to(device)
    best_valid_loss = float('inf')
    output = {'Train_Loss': [], 'Valid_Loss': [], 'Learning_Rate': []}
    for epoch in tqdm(range(num_epochs)):
        start = time.time()
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fun(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        output['Train_Loss'].append(train_loss / len(train_loader))

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                loss = loss_fun(model(inputs), targets)
                valid_loss += loss.item()
        output['Valid_Loss'].append(valid_loss / len(valid_loader))

        scheduler.step()
        output['Learning_Rate'].append(scheduler.get_last_lr()[0])

        if output['Valid_Loss'][-1] < best_valid_loss:
            best_valid_loss = output['Valid_Loss'][-1]
            model.save_pretrained(save_pretrained)
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {output['Train_Loss'][-1]:.8f}, "
              f"Valid Loss: {output['Valid_Loss'][-1]:.8f}, "
              f"Learning Rate: {output['Learning_Rate'][-1]:.6f}, "
              f"Time:{time.time() - start:.4f}s \n")
    pd.DataFrame(output).to_csv(os.path.join(save_pretrained, 'outputs.csv'))