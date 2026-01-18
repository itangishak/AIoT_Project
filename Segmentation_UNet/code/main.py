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
        epoch_loss = 0
        step = 0  # number of mini-batches
        for x, y in dataload:  # iterate over mini-batches
            optimizer.zero_grad()  # zero gradients each mini-batch
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()  # backpropagate
            optimizer.step()  # update parameters
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
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
    model = unet.UNet(3,1)
    model.load_state_dict(torch.load(args.weight,map_location='cpu'))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.normpath(os.path.join(script_dir, '..', 'data'))
    val_dir = os.path.join(base_data_dir, "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Dataset directory not found: {val_dir}. Expected sibling 'data/val' relative to this script")
    liver_dataset = LiverDataset(val_dir, transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset)  # batch_size defaults to 1
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(0.01)
        plt.show()


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
