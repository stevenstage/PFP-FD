import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/.cache/'

import numpy as np
import time
import argparse
import torch
from typing import Tuple
from data_ssl import data_load, data_load_with_cache
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
from mantis.adapters import MultichannelProjector
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from logme import LogME

from utils import Ncc_score, feature_reduce

def parse_args():
    parser = argparse.ArgumentParser(description='Mantis model fine-tuning for CSV electrical fault diagnosis')
    
    # Data parameters
    parser.add_argument('--csv_data_path', type=str, default='/root/autodl-tmp/Renomeado',
                       help='Path to CSV dataset folder')
    parser.add_argument('--train_ratio', type=float, default=0.008, 
                       help='Ratio of training data to use (default: 0.01)')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size for segmentation (default: 512)')
    parser.add_argument('--stride', type=int, default=512,
                       help='Stride for sliding window (default: 256)')
    parser.add_argument('--selected_freqs', type=str, default='all',
                       help='Comma-separated frequencies (e.g., "30,60") or "all"')
    parser.add_argument('--selected_loads', type=str, default='all',
                       help='Comma-separated loads (e.g., "75,100") or "all"')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use cached data if available')
    parser.add_argument('--force_rebuild', action='store_true', default=False,
                       help='Force rebuild dataset cache')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=2e-4, 
                       help='Initial learning rate for AdamW optimizer (default: 2e-4)')
    parser.add_argument('--beta1', type=float, default=0.9, 
                       help='Beta1 parameter for AdamW optimizer (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, 
                       help='Beta2 parameter for AdamW optimizer (default: 0.999)')
    parser.add_argument('--weight_decay', type=float, default=0.1, 
                       help='Weight decay for AdamW optimizer (default: 0.1)')
    
    # Adapter parameters
    parser.add_argument('--new_num_channels', type=int, default=3, 
                       help='Number of output channels for projector (default: 3)')
    parser.add_argument('--patch_window_size', type=int, default=1, 
                       help='Window size for patching (default: 1)')
    parser.add_argument('--base_projector', type=str, default='svd', 
                       choices=['pca', 'svd', 'rand'],
                       help='Projection method: pca, svd, or rand (default: svd)')
    
    # Fine-tuning strategy
    parser.add_argument('--fine_tuning_type', type=str, default='full', 
                       choices=['full', 'scratch', 'adapter_head', 'head'],
                       help='Fine-tuning strategy (default: full)')
    
    # Evaluation parameters
    parser.add_argument('--pca_dim', type=int, default=128,
                       help='PCA dimension for NCC evaluation (default: 128)')
    parser.add_argument('--ncc_divide', type=int, default=8,
                       help='Number of segments for NCC evaluation (default: 8)')
    
    # Output
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results to file')
    parser.add_argument('--results_dir', type=str, default='./mantis_results',
                       help='Directory to save results')
    
    return parser.parse_args()


def parse_list_param(param_str):
    """Parse comma-separated parameter string"""
    if param_str.lower() == 'all':
        return None
    try:
        return [int(x.strip()) for x in param_str.split(',')]
    except ValueError:
        print(f"Warning: Could not parse '{param_str}', using all values")
        return None


def load_and_preprocess_data(args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load CSV data for Mantis fine-tuning
    
    Returns:
        data_train: (N_train, 4, 512) tensor
        y_train: (N_train,) tensor
        data_test: (N_test, 4, 512) tensor
        y_test: (N_test,) tensor
    """
    
    selected_freqs = parse_list_param(args.selected_freqs)
    selected_loads = parse_list_param(args.selected_loads)
    
    print(f"\n{'='*60}")
    print("Loading CSV Electrical Fault Dataset for Mantis")
    print(f"{'='*60}")
    print(f"Data path: {args.csv_data_path}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Window size: {args.window_size}, Stride: {args.stride}")
    print(f"Selected frequencies: {selected_freqs if selected_freqs else 'All'}")
    print(f"Selected loads: {selected_loads if selected_loads else 'All'}")
    print(f"{'='*60}\n")
    
    if args.use_cache:
        test, train = data_load_with_cache(
            train_ratio=args.train_ratio,
            csv_data_path=args.csv_data_path,
            window_size=args.window_size,
            stride=args.stride,
            selected_freqs=selected_freqs,
            selected_loads=selected_loads,
            force_rebuild=args.force_rebuild
        )
    else:
        test, train = data_load(
            train_ratio=args.train_ratio,
            csv_data_path=args.csv_data_path,
            window_size=args.window_size,
            stride=args.stride,
            selected_freqs=selected_freqs,
            selected_loads=selected_loads
        )
    
    # Unpack data
    _, data_train_np, y_train_np = train
    _, data_test_np, y_test_np = test
    
    # Convert to torch tensors
    data_train = torch.tensor(data_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    data_test = torch.tensor(data_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Train: {data_train.shape} (samples, channels, time_steps)")
    print(f"   Train labels: {y_train.shape}, classes: {torch.unique(y_train).tolist()}")
    print(f"   Test: {data_test.shape}")
    print(f"   Test labels: {y_test.shape}, classes: {torch.unique(y_test).tolist()}")
    
    return data_train, y_train, data_test, y_test


def initialize_model(device):
    """Initialize Mantis-8M model"""
    print("\n" + "="*60)
    print("Initializing Mantis-8M Model")
    print("="*60)
    network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
    return MantisTrainer(device=device, network=network)


def evaluate_model(y_true, y_pred, stage="Test"):
    """Comprehensive model evaluation"""
    print(f"\n{'='*60}")
    print(f"{stage} Evaluation Results")
    print(f"{'='*60}")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (macro)")
    print(f"Recall:    {recall:.4f} (macro)")
    print(f"F1-score:  {f1_macro:.4f} (macro)")
    print(f"F1-score:  {f1_micro:.4f} (micro)")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }


def evaluate_ncc(Z_train, y_train, args):
    """Evaluate NCC score"""
    print(f"\n{'='*60}")
    print("NCC Evaluation")
    print(f"{'='*60}")
    
    print(f"Reducing features to {args.pca_dim} dimensions...")
    X_features = feature_reduce(Z_train, args.pca_dim)
    X_features = torch.Tensor(X_features).cuda()
    
    print("Computing SVD...")
    start_time = time.time()
    U, s, VT = torch.linalg.svd(X_features)
    U = U.cpu().numpy()
    s = s.cpu().numpy()
    VT = VT.cpu().numpy()
    end_time = time.time()
    
    divide = args.ncc_divide
    partition_size = args.pca_dim // divide
    sum_s = np.sum(s)
    
    nccscore_list = {}
    ratio_list = {}
    
    for i in range(divide):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i < divide - 1 else args.pca_dim
        
        sub_U = U[:, start_idx:end_idx]
        sub_s = np.diag(s[start_idx:end_idx])
        sub_VT = VT[start_idx:end_idx, :]
        i_features = np.dot(sub_U, np.dot(sub_s, sub_VT))
        
        nccscore_list[i] = float(Ncc_score(i_features, y_train))
        ratio_list[i] = float(np.sum(s[start_idx:end_idx]) / sum_s) if sum_s > 0 else 0.0
    
    weighted_ncc = sum(nccscore_list[i] * ratio_list[i] for i in range(divide))
    
    print(f"NCC segment scores: {nccscore_list}")
    print(f"Singular value ratios: {ratio_list}")
    print(f"Weighted NCC score: {weighted_ncc:.4f}")
    print(f"SVD computation time: {end_time - start_time:.2f} s")
    
    return {
        "time": end_time - start_time,
        "ncc": nccscore_list,
        "ratio": ratio_list,
        "weighted_ncc": weighted_ncc
    }


def save_results(args, metrics, ncc_result, logme_score, training_time):
    """Save results to file"""
    os.makedirs(args.results_dir, exist_ok=True)
    
    freq_str = args.selected_freqs.replace(',', '_') if args.selected_freqs != 'all' else 'all'
    load_str = args.selected_loads.replace(',', '_') if args.selected_loads != 'all' else 'all'
    
    filename = f"mantis_csv_{freq_str}freq_{load_str}load_ratio{args.train_ratio:.4f}.txt"
    filepath = os.path.join(args.results_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Mantis-8M Fine-tuning Results on CSV Dataset\n")
        f.write("="*60 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Data path: {args.csv_data_path}\n")
        f.write(f"  Train ratio: {args.train_ratio}\n")
        f.write(f"  Window size: {args.window_size}, Stride: {args.stride}\n")
        f.write(f"  Selected frequencies: {args.selected_freqs}\n")
        f.write(f"  Selected loads: {args.selected_loads}\n")
        f.write(f"  Fine-tuning type: {args.fine_tuning_type}\n")
        f.write(f"  Projector: {args.base_projector}\n")
        f.write(f"  Epochs: {args.num_epochs}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1 (macro): {metrics['f1_macro']:.4f}\n")
        f.write(f"  F1 (micro): {metrics['f1_micro']:.4f}\n\n")
        
        f.write(f"LogME Score: {logme_score:.4f}\n\n")
        
        f.write("NCC Evaluation:\n")
        f.write(f"  Weighted NCC: {ncc_result['weighted_ncc']:.4f}\n")
        f.write(f"  NCC segments: {ncc_result['ncc']}\n")
        f.write(f"  Singular value ratios: {ncc_result['ratio']}\n\n")
        
        f.write(f"Training time: {training_time:.2f} seconds\n")
    
    print(f"\n✅ Results saved to: {filepath}")


def main():
    args = parse_args()
    
    # Load data
    data_train, y_train, data_test, y_test = load_and_preprocess_data(args)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    try:
        # Initialize model
        model = initialize_model(device)
        
        # Setup adapter for channel projection
        print(f"\n{'='*60}")
        print("Setting up MultichannelProjector Adapter")
        print(f"{'='*60}")
        print(f"Input channels: 4 (CH1-CH4)")
        print(f"Output channels: {args.new_num_channels}")
        print(f"Projection method: {args.base_projector}")
        print(f"Patch window size: {args.patch_window_size}")
        
        adapter = MultichannelProjector(
            new_num_channels=args.new_num_channels,
            patch_window_size=args.patch_window_size,
            base_projector=args.base_projector
        )
        
        print("Fitting adapter on training data...")
        adapter.fit(data_train)
        
        print("Transforming data...")
        X_reduced_train = adapter.transform(data_train)
        X_reduced_test = adapter.transform(data_test)
        
        print(f"✅ Adapter ready!")
        print(f"   Original train shape: {data_train.shape}")
        print(f"   Reduced train shape: {X_reduced_train.shape}")
        print(f"   Original test shape: {data_test.shape}")
        print(f"   Reduced test shape: {X_reduced_test.shape}")
        
        # Setup optimizer
        def init_optimizer(params): 
            return torch.optim.AdamW(
                params, 
                lr=args.learning_rate, 
                betas=(args.beta1, args.beta2), 
                weight_decay=args.weight_decay
            )
        
        # Fine-tune model
        print(f"\n{'='*60}")
        print(f"Fine-tuning Mantis Model ({args.fine_tuning_type})")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_reduced_train)}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Weight decay: {args.weight_decay}")
        
        t0_ft = time.time()
        model.fit(
            X_reduced_train, 
            y_train, 
            num_epochs=args.num_epochs,
            fine_tuning_type=args.fine_tuning_type,
            init_optimizer=init_optimizer
        )
        t1_ft = time.time()
        training_time = t1_ft - t0_ft
        print(f"\n✅ Fine-tuning completed in {training_time:.2f} seconds")
        
        # Extract features
        print(f"\n{'='*60}")
        print("Extracting Features")
        print(f"{'='*60}")
        print("Extracting training features...")
        Z_train_ft = model.transform(X_reduced_train)
        print("Extracting test features...")
        Z_test_ft = model.transform(X_reduced_test)
        print(f"✅ Features extracted!")
        print(f"   Train features shape: {Z_train_ft.shape}")
        print(f"   Test features shape: {Z_test_ft.shape}")
        
        # Evaluate with LogME
        print(f"\n{'='*60}")
        print("LogME Transferability Evaluation")
        print(f"{'='*60}")
        t0_logme = time.time()
        logme = LogME(regression=False)
        logme_score = logme.fit(Z_train_ft, y_train.numpy())
        t1_logme = time.time()
        print(f"✅ LogME score: {logme_score:.4f}")
        print(f"   Evaluation time: {t1_logme - t0_logme:.2f} seconds")
        
        # Evaluate with NCC
        ncc_result = evaluate_ncc(Z_train_ft, y_train, args)
        
        # Make predictions
        print(f"\n{'='*60}")
        print("Making Predictions")
        print(f"{'='*60}")
        y_pred = model.predict(X_reduced_test)
        
        # Convert to numpy
        y_test_np = y_test.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)
        
        # Evaluate model
        metrics = evaluate_model(y_test_np, y_pred_np, stage="Test")
        
        # Save results
        if args.save_results:
            save_results(args, metrics, ncc_result, logme_score, training_time)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset: CSV Electrical Fault Diagnosis")
        print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")
        print(f"Classes: {len(torch.unique(y_train))} (0-6: Normal, HI-1/2/3, LI-1/2/3)")
        print(f"\nPerformance:")
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-score (macro): {metrics['f1_macro']:.4f}")
        print(f"  LogME: {logme_score:.4f}")
        print(f"  Weighted NCC: {ncc_result['weighted_ncc']:.4f}")
        print(f"\nTraining time: {training_time:.2f} seconds")
        print(f"{'='*60}\n")
        
        return metrics, logme_score, ncc_result
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()