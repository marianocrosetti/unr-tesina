"""
Neural network models and data transformations for signature verification.

This module provides ResNet-based encoders, Siamese classifier architecture,
and image transformation pipelines optimized for signature and digit images.
"""

from src.all_deps import *
from src.tools import binarize_image, get_device, get_y_vocab

device = get_device()
lr = 5e-4
adam_eps = 1e-5

def x_transform_signatures():
    """
    Create image transformation pipeline for signature images.
    
    Applies binarization, resizing, normalization, and ImageNet preprocessing
    for use with pretrained ResNet models.
    
    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline.
    """
    transform = transforms.Compose([
        torchvision.transforms.Lambda(PIL.Image.open),
        torchvision.transforms.Lambda(np.asarray),
        torchvision.transforms.Lambda(binarize_image), 
        torchvision.transforms.Lambda(torch.from_numpy),
        torchvision.transforms.Lambda(lambda t : t.unsqueeze(0)), # Add batch dimension
        torchvision.transforms.Resize((122,244)), # Resize all to same size for batching (with rectangular ratio 1x2)
        v2.RGB(), # Transform to RGB so we can use pretrained resnet
        v2.ToDtype(torch.float32, scale=True),
        torchvision.transforms.Lambda(lambda v : 1.0-v.float()), # Negate the image (black: background, white: signature)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize using Imagenet features (same as resnet pretrain)
    ])
    return transform

def y_transform_category(ds):
    """
    Create label transformation that maps string labels to integer indices.
    
    Args:
        ds: Dataset with (x, y) pairs where y is the label.
    
    Returns:
        torchvision.transforms.Lambda: Transform that converts labels to indices.
    """
    itos,stoi = get_y_vocab(ds)
    return torchvision.transforms.Lambda(lambda ys : stoi[ys])

def x_transform_mnist():
    """
    Create image transformation pipeline for MNIST digit images.
    
    Converts to tensor, RGB format, and applies ImageNet normalization
    for use with pretrained ResNet models.
    
    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline.
    """
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        v2.RGB(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
    return transform

class Encoder_Resnet18(torch.nn.Module):
    """
    ResNet-18 based feature encoder for image embeddings.
    
    Uses pretrained ResNet-18 convolutional layers followed by adaptive pooling
    and flattening to produce fixed-size feature vectors.
    
    Args:
        final_pooling (bool): If True, applies adaptive avg and max pooling. Defaults to True.
    """
    def __init__(self, final_pooling=True):
        super().__init__()
        resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.convs = torch.nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4
        )
        self.final_pooling = final_pooling
        if final_pooling:
            self.pooling_avg = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.pooling_max = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.flatten =  torch.nn.Flatten()

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Flattened feature vector of shape (batch_size, feature_dim).
        """
        x = self.convs(x)
        if self.final_pooling:
            ap = self.pooling_avg(x)
            mp = self.pooling_max(x)
            x = torch.stack([ap,mp],dim=-1)
        x = self.flatten(x)
        return x

def mlp(in_features, out_features, hidden_dim = 512):
    """
    Create a multi-layer perceptron with batch normalization and dropout.
    
    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        hidden_dim (int): Hidden layer dimension. Defaults to 512.
    
    Returns:
        torch.nn.Sequential: MLP module with BatchNorm, Dropout, and ReLU activations.
    """
    return torch.nn.Sequential(
        torch.nn.BatchNorm1d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.Dropout(p=0.25),
        torch.nn.Linear(in_features,hidden_dim),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(hidden_dim, out_features),
    )

class SiamessClassifier(torch.nn.Module):
    """
    Siamese network classifier for comparing pairs of images.
    
    Uses a shared encoder to extract features from two images, computes their
    difference (or concatenation), and predicts whether they belong to the same class.
    
    Args:
        encoder (torch.nn.Module): Feature encoder network.
        out_encoder_dim (int): Output dimension of the encoder.
        use_concat (bool): If True, concatenate features; otherwise use difference. Defaults to False.
    """
    def __init__(self, encoder, out_encoder_dim, use_concat = False):
        super().__init__()
        self.encoder = encoder
        self.use_concat = use_concat
        self.head = mlp(out_encoder_dim * (2 if use_concat else 1), 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x1,x2, y1, y2):
        """
        Forward pass through the Siamese classifier.
        
        Args:
            x1: First image batch.
            x2: Second image batch.
            y1: Labels for first image batch.
            y2: Labels for second image batch.
        
        Returns:
            tuple: (f1, f2, labels, logits, loss) containing:
                - f1: Features for first image batch
                - f2: Features for second image batch
                - labels: Binary labels indicating if pairs match (1) or not (0)
                - logits: Raw model predictions
                - loss: Binary cross-entropy loss
        """
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        if self.use_concat:
            features = torch.concat([f1,f2],dim=-1)
        else:
            features = f1-f2
        logits = self.head(features).squeeze(1)
        labels = (y1==y2).float()
        loss = self.criterion(logits, labels)
        return f1, f2, labels, logits, loss