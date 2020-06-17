import numpy as np
import os
from PIL import Image
from torchvision.datasets import VisionDataset


class TinyImageNet_pretrain(VisionDataset):
    base_folder = 'tinyimagenet_pretrain'

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(TinyImageNet_pretrain, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        split = 'train' if train else 'val'

        self.targets = np.load(os.path.join(self.root, self.base_folder, f'{split}_label.npy'))
        self.data = np.load(os.path.join(self.root, self.base_folder, f'{split}_data.npy'))
        # The data is saved in BGR format. Convert to RGB.
        self.data = self.data[...,::-1]


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
    

class TinyImageNet_finetune(VisionDataset):
    base_folder = 'tinyimagenet_finetune'

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(TinyImageNet_finetune, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        split = 'train' if train else 'val'

        self.targets = np.load(os.path.join(self.root, self.base_folder, f'{split}_label.npy'))
        self.data = np.load(os.path.join(self.root, self.base_folder, f'{split}_data.npy'))
        # The data is saved in BGR format. Convert to RGB.
        self.data = self.data[...,::-1]


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
    
class TinyImageNet_finetune5(VisionDataset):
    base_folder = 'tinyimagenet_finetune'

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(TinyImageNet_finetune5, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        split = 'train' if train else 'val'

        self.targets = np.load(os.path.join(self.root, self.base_folder, f'{split}_label.npy'))
        self.data = np.load(os.path.join(self.root, self.base_folder, f'{split}_data.npy'))
        # The data is saved in BGR format. Convert to RGB.
        self.data = self.data[...,::-1]
        
        targets = np.array(self.targets)
        data = np.array(self.data)
        
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(5):
            if self.train:
                sub_cls_id = np.random.choice(np.where(targets==i)[0],125,replace=False)
            else:
                sub_cls_id = np.random.choice(np.where(targets==i)[0],50,replace=False)
                #np.where(ds.targets==i)[0]                
            sub_ds_data_list.append(data[sub_cls_id,:,:,:])
            sub_ds_target_list.append(targets[sub_cls_id])
        
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


    