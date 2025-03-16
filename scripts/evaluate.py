import argparse
import sys
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml.models.classifier import WasteClassifier
from ml.data.dataset import WasteDataset
from ml.utils.metrics import calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate waste classification model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to test data directory')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = WasteClassifier(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = WasteDataset(
        data_dir=args.data_dir,
        transform=config['data']['normalization']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Evaluation
    all_preds = []
    all_labels = []
    
    print('Starting evaluation...')
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate and print metrics
    metrics = calculate_metrics(all_labels, all_preds)
    print('\nEvaluation Results:')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Print confusion matrix
    print('\nConfusion Matrix:')
    print(metrics['confusion_matrix'])

if __name__ == '__main__':
    main() 