import argparse
import sys
import yaml
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml.models.classifier import WasteClassifier

def parse_args():
    parser = argparse.ArgumentParser(description='Predict waste category for images')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    return parser.parse_args()

def load_image(image_path, config):
    """Load and preprocess image for prediction"""
    image = Image.open(image_path).convert('RGB')
    
    # Create transform pipeline
    transform = transforms.Compose([
        transforms.Resize(config['model']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['data']['normalization']['mean'],
            std=config['data']['normalization']['std']
        )
    ])
    
    return transform(image).unsqueeze(0)

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
    
    # Load and preprocess image
    image = load_image(args.image, config)
    image = image.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    # Get class names (assuming they are directory names in the data folder)
    class_names = ['可回收物', '有害垃圾', '厨余垃圾', '其他垃圾', '大件垃圾', '装修垃圾']
    
    # Print results
    predicted_class = class_names[prediction.item()]
    confidence = confidence.item() * 100
    
    print(f'\nPrediction Results for: {Path(args.image).name}')
    print(f'Predicted Class: {predicted_class}')
    print(f'Confidence: {confidence:.2f}%')
    
    # Print top-3 predictions
    top_k = 3
    topk_prob, topk_indices = torch.topk(probabilities, top_k)
    
    print(f'\nTop {top_k} Predictions:')
    for i in range(top_k):
        class_name = class_names[topk_indices[0][i].item()]
        prob = topk_prob[0][i].item() * 100
        print(f'{i+1}. {class_name}: {prob:.2f}%')

if __name__ == '__main__':
    main() 