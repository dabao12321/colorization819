from colorization.colorizers.ade2k_dataset import ADE2kDataset
from colorization.colorizers.util import *
import os
import numpy as np
from PIL import Image
import torch
import time
import sys


def bucket_x_vals(a, b, bucket_size=26, num_buckets=10):
    # bounds checks
    if a < -128 or a > 127:
        print("a is out of range:", a)
    if b < -128 or b > 127:
        print("b is out of range:", b)
    
    a_upper_lim = -128
    b_upper_lim = -128
    a_i = 1
    b_i = 1
    while a_upper_lim < 127:
        if a < a_upper_lim + a_i*bucket_size:
            # buckets are zero indexed
            break
        a_i += 1

    while b_upper_lim < 127:
        if b < b_upper_lim + b_i*bucket_size:
            # buckets are zero indexed
            break
        b_i += 1

    bucket_a = a_i - 1
    bucket_b = b_i - 1

    # return 1d index in a-major order 
    return bucket_a*num_buckets + bucket_b

def get_file_item(index):
    index += 1
    num_zeros = 8 - len(str(index))

    file_prefix = "ADE_train_"
    file_prefix += "0"*num_zeros + str(index)

    img_path = "../data/ADEChallengeData2016/images/training/" + file_prefix + ".jpg"
    mask_path = "../data/ADEChallengeData2016/annotations/training/" + file_prefix + ".png"

    img = Image.open(img_path)
    img = img.convert("RGB")
    mask = Image.open(mask_path)
    
    img_l_orig, img_l_rs, img_ab_rs = preprocess_img_numpy(np.asarray(img), HW=(256,256), resample=Image.NEAREST, ade2k=True)
    mask_rs = mask.resize((256, 256), resample=Image.NEAREST)

    return img_l_rs, img_ab_rs, mask_rs

if __name__ == "__main__":
    
    # goal
    # create new data files for a/b inferred
    # save inferred ab (2d) in numpy .npy files
    # load from .npy in ade2kdataset's getitem (less extra work)

    """
    create map of mask values to ab vals, i.e. mask_val: [(a_val, b_val), ...]

    for each training img:
        1. get the mask from .png file
        2. for each pixel, add ab val to mask_dict 
            - may need to bucket the pixels into 10 buckets each for a, b for space issues
            - then sample the middle value for a,b ?
            - maybe also take into account the L value as part of the dictionary key?
        3. print every 100 ims or so
    
    save the dict somewhere, takes a while to calculate

    for each training img:
        1. get the mask from .png file
        2. for each pixel, randomly sample ab val from corresp dict entry of ab vals

    """
    
    # train_dataset = ADE2kDataset("/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/images/training", \
    #                         "/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/annotations/training", \
    #                         "train")
    # training = 20210, val = 2000

    # mask_to_ab: mask_val --> 1d list of [count_bucket01, count_bucket01, ... count_bucket99]

    mask_to_ab = []
    bucket_val_map = {}

    # Prepopulate mask_to_ab with lists of 0s
    num_mask_classes = 151
    num_buckets = 100
    for i in range(num_mask_classes):
        mask_to_ab.append([0]*num_buckets)
    
    # convert mask_to_ab to numpy
    mask_to_ab = np.asarray(mask_to_ab)

    # Prepopulate bucket_val_map with quicker (a, b) -> bucket_idx val
    for a in range(-128, 128):
        for b in range(-128, 128):
            bucket_val_map[(a, b)] = bucket_x_vals(a, b)
    
    std_dev_sum = 0
    mean_sum = 0

    start_time = time.process_time()
    for file_idx in range(20210):
        try:
            img_l_rs, img_ab_rs, mask_rs = get_file_item(file_idx)
            mask_np = np.asarray(mask_rs)

            # for semseg, calculate mean and std of L channel
            std_dev_sum += np.std(img_l_rs)
            mean_sum += np.mean(img_l_rs)

            # for colorization, add bucketed ab_val to mask_to_ab
            for i in range(256):
                for j in range(256):
                    mask_val = mask_np[i, j]
                    ab_val = (int(img_ab_rs[i, j, 0]), int(img_ab_rs[i, j, 1]))
                    bucket_idx = bucket_val_map[(ab_val)]
                    # mask_to_ab[mask_val][bucket_idx] += 1
                    mask_to_ab[mask_val, bucket_idx] += 1

            if file_idx % 100 == 0:
                print("current file index: ", file_idx)      
                print("\tcurrent std_dev_sum: ", std_dev_sum, " \t mean_sum: ", mean_sum) 
                print("\tcurrent time elapsed", time.process_time() - start_time)
        
        except KeyboardInterrupt:
            print("Keyboard interrupt at ", file_idx)
            print("saving to files...")
            with open("ade2k_statistics.txt", "w") as f:
                f.write("incomplete, broke at index" + str(file_idx) + "\n")
                f.write("std_dev_sum of ade2k: " + str(std_dev_sum) + "\n")
                f.write("mean_sum of ade2k: " + str(mean_sum))
                f.close()

            with open("ade2k_mask_distribution.npy", "wb") as f:
                np.save(f, mask_to_ab)
                f.close()
                
            sys.exit()
            pass
        except Exception as e:
            print("ERROR occurred at index: ", file_idx)
            print(e)
            print("saving to files...")
            with open("ade2k_statistics.txt", "w") as f:
                f.write("incomplete, broke at index" + str(file_idx) + "\n")
                f.write("std_dev_sum of ade2k: " + str(std_dev_sum) + "\n")
                f.write("mean_sum of ade2k: " + str(mean_sum))
                f.close()

            with open("ade2k_mask_distribution.npy", "wb") as f:
                np.save(f, mask_to_ab)
                f.close()
            
    std_dev = std_dev_sum/20210
    mean = mean_sum/20210

    with open("ade2k_statistics.txt", "w") as f:
        f.write("std_dev of ade2k: " + str(std_dev) + "\n")
        f.write("mean of ade2k: " + str(mean))
        f.close()

    with open("ade2k_mask_distribution.npy", "wb") as f:
        np.save(f, mask_to_ab)
        f.close()
