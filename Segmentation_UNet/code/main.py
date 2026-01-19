import torch
from torchvision.transforms import transforms as T
import os
import argparse # argparse parses command-line arguments, e.g., python parseTest.py input.txt --port=8080
import unet
from torch import optim
from dataset import LiverDataset
from torch.utils.data import DataLoader


# Use current CUDA device if available; otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_transform = T.Compose([
    T.ToTensor(),
    # Normalize to [-1, 1] with mean and std
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# Mask only needs conversion to tensor
y_transform = T.ToTensor()

def train_model(model,criterion,optimizer,dataload,num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0.0
        step = 0
        correct = 0
        total = 0
        tp = 0.0
        pred_sum = 0.0
        true_sum = 0.0
        for x, y in dataload:
            optimizer.zero_grad()
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            preds = (outputs > 0.5).float()
            labs = (labels > 0.5).float()
            correct += (preds == labs).sum().item()
            total += labels.numel()
            tp += (preds * labs).sum().item()
            pred_sum += preds.sum().item()
            true_sum += labs.sum().item()
            print("%d/%d,train_loss:%0.3f" % (step, max(1, dataset_size // dataload.batch_size), loss.item()))
        avg_loss = epoch_loss / max(1, step)
        train_acc = correct / total if total > 0 else 0.0
        dice = (2 * tp) / (pred_sum + true_sum + 1e-8) if (pred_sum + true_sum) > 0 else 0.0
        iou = tp / (pred_sum + true_sum - tp + 1e-8) if (pred_sum + true_sum - tp) > 0 else 0.0
        print("epoch %d avg_loss:%0.4f train_acc:%0.4f dice:%0.4f iou:%0.4f" % (epoch, avg_loss, train_acc, dice, iou))
    torch.save(model.state_dict(),'weights_%d.pth' % epoch)  # save model weights for the last epoch
    return model

# Train model
def train():
    model = unet.UNet(3,1).to(device)
    batch_size = args.batch_size
    # loss function
    criterion = torch.nn.BCELoss()
    # optimizer
    optimizer = optim.Adam(model.parameters())  # model.parameters():Returns an iterator over module parameters
    # load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.normpath(os.path.join(script_dir, '..', 'data'))
    train_dir = os.path.join(base_data_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Dataset directory not found: {train_dir}. Expected sibling 'data/train' relative to this script")
    liver_dataset = LiverDataset(train_dir, transform=x_transform, target_transform=y_transform)
    dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    # DataLoader: packages dataset into mini-batches of given batch_size
    # batch_size: how many samples per mini-batch to load
    # shuffle: shuffle dataset each epoch
    # num_workers: number of worker processes to speed up data loading
    train_model(model,criterion,optimizer,dataloader)

# Test
def test():
    model = unet.UNet(3,1).to(device)
    state = torch.load(args.weight, map_location=device)
    model.load_state_dict(state)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.normpath(os.path.join(script_dir, '..', 'data'))
    val_dir = os.path.join(base_data_dir, "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Dataset directory not found: {val_dir}. Expected sibling 'data/val' relative to this script")
    liver_dataset = LiverDataset(val_dir, transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset)  # batch_size defaults to 1
    criterion = torch.nn.BCELoss()
    model.eval()
    total_loss = 0.0
    step = 0
    correct = 0
    total = 0
    tp = 0.0
    pred_sum = 0.0
    true_sum = 0.0
    with torch.no_grad():
        for x, y in dataloaders:
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            step += 1
            preds = (outputs > 0.5).float()
            labs = (labels > 0.5).float()
            correct += (preds == labs).sum().item()
            total += labels.numel()
            tp += (preds * labs).sum().item()
            pred_sum += preds.sum().item()
            true_sum += labs.sum().item()
    avg_loss = total_loss / max(1, step)
    test_acc = correct / total if total > 0 else 0.0
    dice = (2 * tp) / (pred_sum + true_sum + 1e-8) if (pred_sum + true_sum) > 0 else 0.0
    iou = tp / (pred_sum + true_sum - tp + 1e-8) if (pred_sum + true_sum - tp) > 0 else 0.0
    print("Test avg_loss:%0.4f test_acc:%0.4f dice:%0.4f iou:%0.4f" % (avg_loss, test_acc, dice, iou))


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()  # Create an ArgumentParser object
    parser.add_argument('action', type=str, help='train or test')  # Add arguments
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight', type=str, help='the path of the mode weight file')
    args = parser.parse_args()
    
    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()
