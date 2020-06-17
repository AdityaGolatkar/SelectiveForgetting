import numpy as np
import os
from PIL import Image
import torchvision
from torchvision.datasets import VisionDataset
root = os.path.expanduser('~/data')

np.random.seed(0)

class Small_MNIST(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Small_MNIST, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        if self.train:
            ds = torchvision.datasets.MNIST(root=root, train=True, download=True)
            ds.targets=np.array(ds.targets)
            sub_ds_data_list=[]
            sub_ds_target_list=[]
            for i in range(10):
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],20,replace=False)
                sub_ds_data_list.append(ds.data[sub_cls_id,:,:])
                sub_ds_target_list.append(ds.targets[sub_cls_id])
            self.data=np.concatenate(sub_ds_data_list)
            self.targets=np.concatenate(sub_ds_target_list)
                
        else:
            ds = torchvision.datasets.MNIST(root=root, train=False, download=True)
            self.data=ds.data.numpy()
            self.targets=ds.targets.numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
    
class Small_Binary_MNIST(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Small_Binary_MNIST, self).__init__(root,transform=transform,target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.MNIST(root=root, train=True, download=True)
        ds.targets=np.array(ds.targets)
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(2):
            if self.train:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],200,replace=False)
            else:
                sub_cls_id = np.where(ds.targets==i)[0]
            sub_ds_data_list.append(ds.data[sub_cls_id,:,:])
            sub_ds_target_list.append(ds.targets[sub_cls_id])
        self.data=np.concatenate(sub_ds_data_list)
        self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
