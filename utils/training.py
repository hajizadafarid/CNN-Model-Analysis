# utils/training.py

import torch
from tqdm import tqdm
import time

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: Average training loss and training accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Evaluation DataLoader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: Average evaluation loss and evaluation accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_and_eval(model, EPOCHS, optimizer, criterion, train_loader, eval_loader, device):
    """
    Train and evaluate the model, returning the best training and evaluation accuracies and total training time.

    Args:
        model (nn.Module): The model to train.
        EPOCHS (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        train_loader (DataLoader): Training DataLoader.
        eval_loader (DataLoader): Evaluation DataLoader.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: Best training accuracy, best evaluation accuracy, total training time.
    """
    train_acc_list = []
    train_loss_list = []
    eval_acc_list = []
    eval_loss_list = []
    print("Training and evaluating...")
    total_time = 0

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        one_epoch_time = time.perf_counter() - start
        total_time += one_epoch_time

        # Evaluate on the validation set
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)

        # Record metrics
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        eval_acc_list.append(eval_acc)
        eval_loss_list.append(eval_loss)

        # Print epoch summary
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Evaluation Loss: {eval_loss:.4f}")
        print(f"Evaluation Accuracy: {eval_acc:.2f}%")
        print("-" * 50)

    # Return the best training and evaluation accuracies along with total training time
    return max(train_acc_list), max(eval_acc_list), total_time

def train_and_eval_with_loss_and_acc(model, EPOCHS, optimizer, criterion, train_loader, eval_loader, device):
    """
    Train and evaluate the model, returning lists of losses and accuracies along with total training time.

    Args:
        model (nn.Module): The model to train.
        EPOCHS (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        train_loader (DataLoader): Training DataLoader.
        eval_loader (DataLoader): Evaluation DataLoader.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: Lists of training losses, evaluation losses, training accuracies, evaluation accuracies, total training time.
    """
    train_acc_list = []
    train_loss_list = []
    eval_acc_list = []
    eval_loss_list = []
    print("Training and evaluating...")
    total_time = 0

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        one_epoch_time = time.perf_counter() - start
        total_time += one_epoch_time

        # Evaluate on the validation set
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)

        # Record metrics
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        eval_acc_list.append(eval_acc)
        eval_loss_list.append(eval_loss)

        # Print epoch summary
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Evaluation Loss: {eval_loss:.4f}")
        print(f"Evaluation Accuracy: {eval_acc:.2f}%")
        print("-" * 50)

    # Return all recorded metrics and total training time
    return train_loss_list, eval_loss_list, train_acc_list, eval_acc_list, total_time
