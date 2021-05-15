import torch
import os, sys
from PIL import Image
from skimage import color
from colorization.colorizers.util import *

class ADE2kDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, split):
        """
        Pass in the directory of images and masks
        train = True if train, false if val
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.split = split

    def __getitem__(self, index):
        """
        Convert index (0 indexed) into filename (1 indexed)
        """
        file_prefix = "ADE_"
        if self.split == "train":
            file_prefix += "train_"
        elif self.split == "val":
            file_prefix += "val_"
        else:
            print("invalid split")
            return
        
        # files are named in 1-indexed
        index += 1
        num_zeros = 8 - len(str(index))
        file_prefix += "0"*num_zeros + str(index)

        img_path = self.img_dir + "/" + file_prefix + ".jpg"
        mask_path = self.mask_dir + "/" + file_prefix + ".png"

        # print("img path", img_path)
        # print("mask path", mask_path)

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img_l_orig, img_l_rs, img_ab_rs = preprocess_img(np.asarray(img), HW=(256,256), resample=Image.NEAREST, ade2k=True)
        mask_rs = mask.resize((256, 256), resample=Image.NEAREST)

        # need to output 3 things (2 things for now)
        # 1. ade2k image bw resized
        # 2. ade2k mask resized, ignore for now
        # 3. ade2k 1 x 2 x 256 x 256 image ab tensor

        return img_l_rs, img_ab_rs
        


if __name__=="__main__":
    # print(sys.path)
    my_data = ADE2kDataset("/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/images/training", \
                            "/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/annotations/training", \
                            "train")
   
   

        