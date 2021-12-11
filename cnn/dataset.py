import os, shutil
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
import csv
import torch
import torchvision
import torchvision.transforms as transforms


def transfer_data():
    labels_dict = {}
    with open("trainval/labels.csv") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row[1] != "label":
                labels_dict[row[0]] = row[1]
    img_files = glob('trainval/*/*_image.jpg')
    img_label_dict = {"0":[],"1":[],"2":[]}
    for i in tqdm(range(len(img_files))):
        file_name = img_files[i].replace("trainval/","")
        file_name = file_name.replace("_image.jpg","")
        label = labels_dict[file_name]
        img_label_dict[label].append(img_files[i])
        pass
    assert(len(img_label_dict["0"])+len(img_label_dict["1"])+len(img_label_dict["2"])==len(img_files))
    print("finish classification")
    for label in img_label_dict.keys():
        dest_folder = "data/trainval/"+label
        file_list = img_label_dict[label]
        for i in range(len(file_list)):
            shutil.copy(file_list[i], dest_folder)
            dst_file = os.path.join(dest_folder, file_list[i].split("/")[-1])
            new_dst_file_name = os.path.join(dest_folder, str(i)+".jpg")
            os.rename(dst_file, new_dst_file_name)
    print("finish copy")

def transfer_data_test():
    img_files = glob('test/*/*_image.jpg')
    with open("result.csv","a+") as f:
        csv_writer = csv.writer(f)
        for i in tqdm(range(len(img_files))):
            dest_folder = "data/test/0"
            f_name = img_files[i].replace("test/", "")
            f_name = f_name.replace("_image.jpg", "")
            csv_writer.writerow([f_name])
            shutil.copy(img_files[i], dest_folder)
            dst_file = os.path.join(dest_folder, img_files[i].split("/")[-1])
            new_dst_file_name = os.path.join(dest_folder, str(i).zfill(4)+".jpg")
            os.rename(dst_file, new_dst_file_name)
    


class VehicleDataset:
    """
    Vehicle Dataset.
    """
    def __init__(self, batch_size=4, dataset_path='/home/mtl-admin/Haojie/project/data'):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.train_dataset = self.get_train_numpy()
        self.x_mean, self.x_std = self.compute_train_statistics()
        self.transform = self.get_transforms()
        self.train_loader, self.val_loader = self.get_dataloaders()

    def get_train_numpy(self):
        trans = transforms.Compose([transforms.Resize((64,116))])
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'trainval'), transform=trans)
        train_x = np.zeros((len(train_dataset), 64, 116, 3))
        for i, (img, _) in enumerate(train_dataset):
            train_x[i] = img
        return train_x / 255.0

    def compute_train_statistics(self):
        x_mean = np.mean(self.train_dataset, axis = (0,1,2))  # per-channel mean
        x_std = np.std(self.train_dataset, axis = (0,1,2))  # per-channel std
        return x_mean, x_std

    def get_transforms(self):
        transform_list = [
            # resize the image to 32x32x3
            transforms.Resize(32),
            # convert image to PyTorch tensor
            transforms.ToTensor(),
            # normalize the image (use self.x_mean and self.x_std)
            transforms.Normalize(self.x_mean, self.x_std),
        ]
        transform = transforms.Compose(transform_list)
        return transform

    def get_dataloaders(self):
        # train set
        train_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'trainval'), transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        # valication set
        val_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'test'), transform=self.transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def plot_image(self, image, label):
        image = np.transpose(image.numpy(), (1, 2, 0))
        image = image * self.x_std.reshape(1, 1, 3) + self.x_mean.reshape(1, 1, 3)  # un-normalize
        plt.title(label)
        plt.imshow(image)
        plt.show()

    def get_semantic_label(self, label):
        return str(label)


if __name__ == '__main__':
    dataset = VehicleDataset()
    print(dataset.x_mean, dataset.x_std)
    images, labels = iter(dataset.val_loader).next()
    dataset.plot_image(
        torchvision.utils.make_grid(images),
        ', '.join([dataset.get_semantic_label(label.item()) for label in labels])
    )
