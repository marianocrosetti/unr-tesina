"""Deep learning helper functions for training and evaluation."""
import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Utilities ---

def shapes(tensors):
    """Get shapes of a list of tensors."""
    return [x.shape for x in tensors]

def flatten_check(inp, targ):
    """Flatten tensors and check they have same number of elements."""
    inp, targ = inp.view(-1), targ.view(-1)
    assert inp.numel() == targ.numel(), f"Mismatch: {inp.numel()} vs {targ.numel()}"
    return inp, targ

def single_batch(dl, device):
    """Get a single batch from dataloader and move to device."""
    b = next(iter(dl))
    b = [x.to(device) for x in b]
    print(f'Getting single batch. Shapes: {shapes(b)}')
    return b

def get_encoder_out_dim(encoder, x):
    """Get output dimension of encoder given sample input."""
    with torch.no_grad():
        return encoder(x).shape[-1]

# --- Models ---

class SiameseContrastive(nn.Module):
    """Siamese network with cosine embedding loss."""
    def __init__(self, encoder, margin=0.5):
        super().__init__()
        self.encoder = encoder
        self.criterion = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, x1, x2, y1, y2):
        f1, f2 = self.encoder(x1), self.encoder(x2)
        labels = torch.where((y1 == y2), 1, -1)
        loss = self.criterion(f1, f2, labels)
        return f1, f2, labels, loss

# --- Metrics and Loss Functions ---

def accuracy_dist(inp, targ, thresh=1):
    """Compute accuracy based on distance predictions.
    Returns fraction of predictions where (distance < thresh) matches target boolean."""
    inp, targ = flatten_check(inp, targ)
    return ((inp < thresh) == targ.bool()).float().mean()

def my_loss_func_LeCun(dist, target, margin=2, reduction='mean'):
    """LeCun's contrastive loss for siamese networks.
    For positive pairs: penalizes large distances.
    For negative pairs: penalizes distances smaller than margin."""
    assert reduction in ['mean', 'none']
    neg_dist = torch.clamp(margin - dist, min=0.0)
    res = torch.where(target.bool(), dist, -neg_dist).pow(2)
    if reduction == 'mean':
        res = res.mean()
    return 0.5 * res

def thresh_finder(preds, targs, acc, x0, xf):
    """Plot accuracy vs threshold to find optimal decision boundary."""
    xs = torch.linspace(x0, xf)
    accs = [acc(preds, targs, thresh=x) for x in xs]
    plt.plot(xs, accs)

# --- Training Loops ---

def training_epoch(model, opt, dl, device, n_epoch=0, num_epochs=1):
    """Run one training epoch for siamese model with (x1, x2, y1, y2) batches."""
    training_loss = 0.0
    model.train()
    for x1, x2, y1, y2 in tqdm.tqdm(dl, desc=f'Training epoch [{n_epoch + 1}/{num_epochs}]'):
        x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device, dtype=torch.float32), y2.to(device, dtype=torch.float32)
        B = x1.size(0)
        opt.zero_grad()
        loss = model(x1, x2, y1, y2)[-1]
        loss.backward()
        opt.step()
        training_loss += loss.item() * B
    epoch_loss = training_loss / len(dl.dataset)
    print(f"Epoch [{n_epoch + 1}/{num_epochs}], Training loss: {epoch_loss:.4f}")

def valid_epoch(model, opt, dl, device, n_epoch=0, num_epochs=1):
    """Run one validation epoch, returns logits and prints loss/accuracy."""
    model.eval()
    val_loss = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x1, x2, y1, y2 in tqdm.tqdm(dl, desc=f'Valid epoch [{n_epoch + 1}/{num_epochs}]'):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device, dtype=torch.float32), y2.to(device, dtype=torch.float32)
            B = x1.size(0)
            logits, loss = model(x1, x2, y1, y2)[-2:]
            all_logits.append(logits)
            all_labels.append(y1 == y2)
            val_loss += loss.item() * B
    all_logits, all_labels = torch.concat(all_logits), torch.concat(all_labels)
    avg_val_loss = val_loss / len(dl.dataset)
    accuracy = ((all_logits.sigmoid() > 0.5) == all_labels).float().mean().item()
    print(f"Epoch [{n_epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return all_logits

def run_training(model, dl_train, dl_valid, device, num_epochs, lr, adam_eps=1e-8):
    """Full training loop with AdamW optimizer."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, eps=adam_eps)
    for n_epoch in range(num_epochs):
        training_epoch(model, opt, dl_train, device, n_epoch, num_epochs)
        valid_epoch(model, opt, dl_valid, device, n_epoch, num_epochs)

def train_classifier_mnist(dl, model, criterion, opt, device):
    """Train binary classifier on MNIST (digit < 5 vs >= 5)."""
    training_loss = 0.0
    model.train()
    for x, y in tqdm.tqdm(dl, desc='Training'):
        x, y = x.to(device), (y < 5).to(device, dtype=torch.float32).unsqueeze(1)
        batch_size = x.size(0)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        training_loss += loss.item() * batch_size
    epoch_loss = training_loss / len(dl.dataset)
    print(f"Training loss: {epoch_loss:.4f}")

def valid_epoch_cosine(model, dl, device, thresh=0.5):
    """Validation epoch using cosine similarity for siamese contrastive models."""
    all_f1, all_f2, all_y1, all_y2 = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x1, x2, y1, y2 in tqdm.tqdm(dl, desc='Validation'):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            f1, f2 = model(x1, x2, y1, y2)[:2]
            all_f1.append(f1)
            all_f2.append(f2)
            all_y1.append(y1)
            all_y2.append(y2)
    all_f1, all_f2 = torch.cat(all_f1), torch.cat(all_f2)
    all_y1, all_y2 = torch.cat(all_y1), torch.cat(all_y2)
    cos = nn.CosineSimilarity(dim=1)
    similarities = cos(all_f1, all_f2)
    predictions = similarities >= thresh
    labels = (all_y1 == all_y2)
    accuracy = (labels == predictions).float().mean()
    print(f"Validation Accuracy (thresh={thresh}): {accuracy:.4f}")
    return similarities, labels, accuracy

