import torch
from tqdm import tqdm


def train(train_loader, model, criterion, sensing_rate, optimizer, device):
    model.train()
    sum_loss = 0
    
    train_bar = tqdm(train_loader, desc='Training', leave=False, dynamic_ncols=True)
    
    for inputs in train_bar:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs, sys_loss = model(inputs)
        gamma1 = torch.Tensor([0.0001]).to(device)
        I = torch.eye(int(sensing_rate * 1024)).to(device)
        loss = criterion(outputs, inputs) + torch.mul(criterion(sys_loss, I), gamma1)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        
        train_bar.set_postfix({'batch_loss': f'{loss.item():.6f}'})
    
    train_bar.close()
    
    return sum_loss



