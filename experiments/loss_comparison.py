import torch.optim as optim
from models.simplemodel import HCNN
from utils.training import train_and_eval
from utils.dataloader import set_all_seeds
from torch import nn
import torch.nn.functional as F
import os
import torch

class SquaredHingeLoss(nn.Module):
    def __init__(self):
        super(SquaredHingeLoss, self).__init__()

    def forward(self, outputs, targets):
        """
        Compute the Squared Hinge Loss.

        Args:
            outputs (torch.Tensor): Output logits from the model.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        num_classes = outputs.size(1)
        one_hot_targets = F.one_hot(targets, num_classes).float()
        
        # Get the score for correct classes
        correct_scores = torch.sum(outputs * one_hot_targets, dim=1)
        
        # Calculate margins for all classes
        margins = outputs - correct_scores.unsqueeze(1) + 1.0  # Add 1 for margin
        
        # Square the positive margins
        squared_margins = torch.pow(torch.clamp(margins, min=0), 2)
        
        # Average over classes and batch
        loss = torch.mean(torch.sum(squared_margins, dim=1))
        
        return loss

def compare_losses(train_loader, eval_loader, device, results_dir='results'):
    """
    Compare different loss functions: CrossEntropyLoss vs SquaredHingeLoss.

    Args:
        train_loader (DataLoader): Training DataLoader.
        eval_loader (DataLoader): Evaluation DataLoader.
        device (torch.device): Device to run computations on.
        results_dir (str, optional): Directory to save results. Defaults to 'results'.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'accuracy_comparison_by_loss_functions.txt')
    
    EPOCHS = 50
    best_num_layers = 5
    best_kernel_size = 3
    learning_rate = 0.001
    criterion_ce = nn.CrossEntropyLoss()
    criterion_hinge = SquaredHingeLoss()
    
    with open(results_file, 'w') as f:
        f.write("Accuracy Comparison:\n")
        
        # Train with CrossEntropyLoss
        set_all_seeds(42)
        model_ce = HCNN(num_of_conv_layers=best_num_layers, 
                       kernel_size=best_kernel_size, 
                       dropout=True, 
                       dropout_fc=0.1).to(device)
        optimizer_ce = optim.SGD(model_ce.parameters(), lr=learning_rate)
        ce_train_acc, ce_eval_acc, _ = train_and_eval(model_ce, EPOCHS, optimizer_ce, criterion_ce, train_loader, eval_loader, device)
        
        # Train with SquaredHingeLoss
        set_all_seeds(42)
        model_hinge = HCNN(num_of_conv_layers=best_num_layers, 
                          kernel_size=best_kernel_size, 
                          dropout=True, 
                          dropout_fc=0.1).to(device)
        optimizer_hinge = optim.Adam(model_hinge.parameters(), lr=learning_rate)
        hinge_train_acc, hinge_eval_acc, _ = train_and_eval(model_hinge, EPOCHS, optimizer_hinge, criterion_hinge, train_loader, eval_loader, device)
        
        # Write comparison to file
        f.write("\nCrossEntropyLoss:\n")
        f.write(f"Train Acc: {ce_train_acc:.2f}%, Val Acc: {ce_eval_acc:.2f}%\n")
        
        f.write("SquaredHingeLoss:\n")
        f.write(f"Train Acc: {hinge_train_acc:.2f}%, Val Acc: {hinge_eval_acc:.2f}%\n")
