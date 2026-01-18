import torch.utils.data as data
import os
import PIL.Image as Image

# data.Dataset:
# All subclasses should override __len__ and __getitem__. The former returns the size of the dataset,
# and the latter supports integer indexing from 0 to len(self)-1.

class LiverDataset(data.Dataset):
    # When creating an instance of LiverDataset, __init__ is called to initialize.
    def __init__(self,root,transform = None,target_transform = None): # root is the image directory
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root directory not found: {root}")
        n = len(os.listdir(root))//2  # number of pairs (image and mask). Uses integer division.
        
        imgs = []
        for i in range(n):
            img = os.path.join(root,"%03d.png"%i)  # join directory and filename
            mask = os.path.join(root,"%03d_mask.png"%i)
            imgs.append([img,mask])  # append the pair [image, mask]
        
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    
    def __getitem__(self,index):
        x_path,y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x,img_y  # return the tensors
    
    
    def __len__(self):
        return len(self.imgs)  # number of image-mask pairs
