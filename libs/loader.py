import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageCms
import os
from sklearn import preprocessing
import pandas as pd

class DatasetGenerator(Dataset):
    def __init__(self, data_dir, list_file, transform1=None, transform2=None, n_class=3, set_name='train_preprocessed'):

        self.data_dir = data_dir
        self.list_file = list_file
        self.transform1 = transform1
        self.transform2 = transform2
        self.n_class = n_class
        self.set_name = set_name

        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")

        self.image_names, self.labels = self.load_eyeQ_excel()

    def load_eyeQ_excel(self):
        image_names = []
        labels = []
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.array(range(self.n_class)))
        df_tmp = pd.read_csv(self.list_file)
        img_num = len(df_tmp)

        for idx in range(img_num):
            image_name = df_tmp["image"][idx]
            image_names.append(os.path.join(self.data_dir, image_name[:-5] + '.png'))

            label = lb.transform([int(df_tmp["quality"][idx])])
            labels.append(label)

        return image_names, labels

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')

        if self.transform1 is not None:
            image = self.transform1(image)

        img_hsv = image.convert("HSV")
        img_lab = ImageCms.applyTransform(image, self.rgb2lab_transform)

        img_rgb = np.asarray(image).astype('float32')
        img_hsv = np.asarray(img_hsv).astype('float32')
        img_lab = np.asarray(img_lab).astype('float32')

        if self.transform2 is not None:
            img_rgb = self.transform2(img_rgb)
            img_hsv = self.transform2(img_hsv)
            img_lab = self.transform2(img_lab)

        if self.set_name == 'train' or self.set_name == 'val':
            label = self.labels[index]
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab), torch.FloatTensor(label)
        else:
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab)

    def __len__(self):
        return len(self.image_names)
