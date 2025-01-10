import argparse
import os
from utils.dataloader import create_train_eval_dataloaders, set_all_seeds
from models.simplemodel import HCNN
from models.pretrainedmodel import PretrainedModel
from experiments.hyperparameter_tuning import (
    checking_num_layers_simplemodel,
    checking_kernel_sizes_simplemodel,
    checking_skip_connections_simplemodel,
    checking_normalisation_simplemodel,
    checking_dropout_simplemodel,
    checking_L2_regularization_simplemodel,
    checking_learning_rates_simplemodel,
    checking_batch_sizes_simplemodel,
    checking_optimizer_types_simplemodel,
    run_all_hyperparameter_tuning
)
from experiments.loss_comparison import compare_losses
from experiments.compression_analysis import analyze_svd_compression
from experiments.model_comparison import compare_models
import torch

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seeds for reproducibility
    set_all_seeds(42)
    
    # Create data loaders
    train_loader, eval_loader = create_train_eval_dataloaders(
        root_dir="./data/101_ObjectCategories",
        train_samples_per_class=15,
        eval_samples_per_class=15,
        batch_size=32
    )
    
    # Execute experiments based on user input
    if args.experiment == 'num_layers':
        checking_num_layers_simplemodel(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'kernel_sizes':
        checking_kernel_sizes_simplemodel(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'skip_connections':
        checking_skip_connections_simplemodel(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'normalisation':
        checking_normalisation_simplemodel(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'dropout':
        checking_dropout_simplemodel(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'weight_decay':
        checking_L2_regularization_simplemodel(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'learning_rates':
        checking_learning_rates_simplemodel(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'batch_sizes':
        # Pass data loader functions to dynamically create loaders with different batch sizes
        checking_batch_sizes_simplemodel(
            train_loader_fn=lambda batch_size: create_train_eval_dataloaders(
                root_dir="./data/101_ObjectCategories",
                train_samples_per_class=15,
                eval_samples_per_class=15,
                batch_size=batch_size
            ),
            eval_loader_fn=lambda batch_size: create_train_eval_dataloaders(
                root_dir="./data/101_ObjectCategories",
                train_samples_per_class=15,
                eval_samples_per_class=15,
                batch_size=batch_size
            ),
            device=device,
            results_dir=args.results_dir
        )
    elif args.experiment == 'optimizer_types':
        checking_optimizer_types_simplemodel(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'hyperparameter_tuning':
        run_all_hyperparameter_tuning(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'loss_comparison':
        compare_losses(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'compression_analysis':
        analyze_svd_compression(train_loader, eval_loader, device, results_dir=args.results_dir)
    elif args.experiment == 'model_comparison':
        # Initialize models
        scratch_model = HCNN(num_of_conv_layers=5, kernel_size=3, dropout=True, dropout_fc=0.1).to(device)
        pretrained_model = PretrainedModel(num_classes=10, use_pretrained=True).to(device)
        
        # Compare models
        compare_models(
            scratch_model, pretrained_model, train_loader, eval_loader, 
            EPOCHS=50, device=device, learning_rate=0.001
        )
    else:
        print("Please specify a valid experiment type. Use --help for more information.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HCNN Experiments')
    parser.add_argument(
        '--experiment', 
        type=str, 
        required=True, 
        help='Type of experiment to run. Options include: num_layers, kernel_sizes, skip_connections, normalisation, dropout, weight_decay, learning_rates, batch_sizes, optimizer_types, hyperparameter_tuning, loss_comparison, compression_analysis, model_comparison'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save experiment results'
    )
    args = parser.parse_args()
    main(args)
