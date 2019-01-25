import PIL
import numpy as np
import sys
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import cv2
from source_target_transforms import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_batches', type=int, default=15000, \
        help='Number of batches to run')
    parser.add_argument('--crop', type=int, default=128, \
        help='Random crop size')
    parser.add_argument('--lr', type=float, default=0.00001, \
        help='Base learning rate for Adam')
    parser.add_argument('--factor', type=int, default=2, \
        help='Interpolation factor.')
    parser.add_argument('--img', type=str, help='Path to input img')

    args = parser.parse_args()

    return args


class DataSampler:
    def __init__(self, img, sr_factor, crop_size):
        self.img = img
        self.sr_factor = sr_factor
        self.pairs = self.create_hr_lr_pairs()
        sizes = np.float32([x[0].size[0]*x[0].size[1] / float(img.size[0]*img.size[1]) \
            for x in self.pairs])
        self.pair_probabilities = sizes / np.sum(sizes)

        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(crop_size),
            ToTensor()])

    def create_hr_lr_pairs(self):
        smaller_side = min(self.img.size[0 : 2])
        larger_side = max(self.img.size[0 : 2])

        factors = []
        for i in range(smaller_side//5, smaller_side+1):
            downsampled_smaller_side = i
            zoom = float(downsampled_smaller_side)/smaller_side
            downsampled_larger_side = round(larger_side*zoom)
            if downsampled_smaller_side%self.sr_factor==0 and \
                downsampled_larger_side%self.sr_factor==0:
                factors.append(zoom)

        pairs = []

        print(factors)
        for zoom in factors:
            hr = self.img.resize((int(self.img.size[0]*zoom), \
                                int(self.img.size[1]*zoom)), \
                resample=PIL.Image.BICUBIC)


            lr = hr.resize((int(hr.size[0]/self.sr_factor), \
                int(hr.size[1]/self.sr_factor)),
                resample=PIL.Image.BICUBIC)
            lr = lr.resize(hr.size, resample=PIL.Image.BICUBIC)
            pairs.append((hr, lr))

        return pairs

    def generate_data(self):
        num = 0
        while True:

            hr, lr = random.choices(self.pairs, weights=self.pair_probabilities, k=1)[0]
            hr_tensor, lr_tensor = self.transform((hr, lr))

            if num%100 == 0:
                fl = './HR images/' + str(num).zfill(4) + '.png'
                hr.save(fl)
                fl =  './LR images/' + str(num).zfill(4) + 'x4.png'
                lr.save(fl)

            hr_tensor = F.to_tensor(hr)
            lr_tensor = F.to_tensor(lr)

            hr_tensor = torch.unsqueeze(hr_tensor, 0)
            lr_tensor = torch.unsqueeze(lr_tensor, 0)
            yield hr_tensor, lr_tensor
        num = num + 1

if __name__ == '__main__':
    args = get_args()

    num = 0
    img = PIL.Image.open(sys.argv[1])
    sampler = DataSampler(img, 2)
    for x in sampler.generate_data():
        hr, lr = x
        hr = hr.numpy().transpose((1, 2, 0))
        lr = lr.numpy().transpose((1, 2, 0))
