# train_mantis_ts.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/.cache/'

import numpy as np
import time
import argparse
import torch
from typing import Tuple
from d import data_load_d  # ✅ 直接导入你的数据加载器
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
from mantis.adapters import MultichannelProjector
from sklearn.metrics import classification_report, f1_score


def parse_args():
    parser = argparse.ArgumentParser(description='Mantis model training script for time series')
    parser.add_argument('--train_ratio', type=float, default=0.24, help='Ratio of training data to use')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Initial learning rate for AdamW optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 parameter for AdamW optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for AdamW optimizer')
    parser.add_argument('--new_num_channels', type=int, default=10, help='Number of output channels for projector (建议=输入通道数)')
    parser.add_argument('--patch_window_size', type=int, default=1, help='Window size for patching (如10表示每10个时间点打包)')
    parser.add_argument('--base_projector', type=str, default='svd', help='choose pca or svd or rand')
    parser.add_argument('--fine_tuning_type', type=str, default='head', help='choose full or scratch or adapter_head or head')
    return parser.parse_args()


def load_and_preprocess_data(train_ratio: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    加载数据并转换为 PyTorch Tensor
    - 输入: (B, 510, 10) 时间序列
    - 输出: 四个 Tensor
    """
    (X_train, y_train), (X_test, y_test) = data_load_d(train_ratio)

    # 转为 Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(f"Train data shape: {X_train.shape}")  # (N, 510, 10)
    print(f"Test data shape: {X_test.shape}")
    print(f"Labels: {y_train.shape}, {y_test.shape}")

    return X_train, y_train, X_test, y_test


def initialize_model(device):
    network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
    return MantisTrainer(device=device, network=network)


def evaluate_model(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    for avg_type in ['macro', 'micro']:
        f1 = f1_score(y_true, y_pred, average=avg_type)
        print(f'{avg_type.capitalize()} F1-score: {f1:.4f}')
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(5)]))


def main():
    args = parse_args()
    X_train, y_train, X_test, y_test = load_and_preprocess_data(args.train_ratio)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model = initialize_model(device)
        
        # 初始化投影器（适配你的数据：10通道时间序列）
        adapter = MultichannelProjector(
            new_num_channels=args.new_num_channels,      # 建议保持10
            patch_window_size=args.patch_window_size,    # 如10，把510变成51个patch
            base_projector=args.base_projector           # svd/pca/rand
        )
        
        # 注意：adapter.fit 需要 (B, T, C) 格式 —— 你数据正好是这个格式
        adapter.fit(X_train)
        X_reduced_train = adapter.transform(X_train)
        X_reduced_test = adapter.transform(X_test)
        
        print(f"Original train dims: {X_train.shape}")      # e.g. (N, 510, 10)
        print(f"Reduced train dims: {X_reduced_train.shape}") # e.g. (N, 51, 10) if patch=10
        
        def init_optimizer(params): 
            return torch.optim.AdamW(
                params, 
                lr=args.learning_rate, 
                betas=(args.beta1, args.beta2), 
                weight_decay=args.weight_decay
            )
            
        t0_ft = time.time()
        model.fit(
            X_reduced_train, 
            y_train, 
            num_epochs=args.num_epochs,
            fine_tuning_type=args.fine_tuning_type,  # 推荐 'head' 或 'adapter_head'
            init_optimizer=init_optimizer
        )
        t1_ft = time.time()
        print(f"Fine-tuning finished in {t1_ft - t0_ft:.2f} s")
        
        # 预测
        y_pred = model.predict(X_reduced_test)
        y_test_np = y_test.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)
        
        evaluate_model(y_test_np, y_pred_np)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error occurred: {str(e)}")


if __name__ == '__main__':
    main()