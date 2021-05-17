from colorization.colorizers.ade2k_dataset import ADE2kDataset
from colorization.colorizers.util import *
import os
import numpy as np
from PIL import Image
import torch
import time
import sys
from joblib import Parallel, delayed

def bucket_to_ab_range(bucket_idx, num_buckets = 10, bucket_size = 26):
    bucket_a = bucket_idx // 10
    bucket_b = bucket_idx % 10
    
    if -128 + (bucket_a + 1) * bucket_size >= 127:
        range_a = (-128 + bucket_a * bucket_size, 128)
    else:
        range_a = (-128 + bucket_a * bucket_size, -128 + (bucket_a + 1) * bucket_size)

    if -128 + (bucket_b + 1) * bucket_size >= 127:
        range_b = (-128 + bucket_b * bucket_size, 128)
    else:
        range_b = (-128 + bucket_b * bucket_size, -128 + (bucket_b + 1) * bucket_size)

    return range_a, range_b

def get_file_mask(index, split="train"):
    index += 1
    num_zeros = 8 - len(str(index))

    if split == "train":
        file_prefix = "ADE_train_"
        file_prefix += "0"*num_zeros + str(index)
        mask_path = "/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/annotations/training/" + file_prefix + ".png"
    elif split == "val":
        file_prefix = "ADE_val_"
        file_prefix += "0"*num_zeros + str(index)
        mask_path = "/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/annotations/validation/" + file_prefix + ".png"

    mask = Image.open(mask_path)
    mask_rs = mask.resize((256, 256), resample=Image.NEAREST)
    return mask_rs

def save_filename(index, split="train"):
    index += 1
    num_zeros = 8 - len(str(index))

    file_prefix = "ADE_inferred_ab_"
    file_prefix += "0"*num_zeros + str(index)

    if split == "train":
        save_path = "/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/inferred_ab/training/" + file_prefix + ".npy"
    elif split == "val":
        save_path = "/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/inferred_ab/validation/" + file_prefix + ".npy"
    return save_path

def infer_ab_npy(file_idx, mask_abval_probs, start_time, split_type = "train", \
                num_buckets = 100, bucket_size = 26):
    # load mask
    mask_np = np.asarray(get_file_mask(file_idx, split=split_type))

    # store final inferred mask in 1 x 2 x 256 x 256 np arr        
    inferred_a = np.zeros((256, 256))
    inferred_b = np.zeros((256, 256))

    # super slow code, 2.9 secs per image
    # for i in range(256):
    #     for j in range(256):
    #         mask_val = mask_np[i, j]
    #         curr_probs = mask_abval_probs[mask_val]
    #         random_bucket = rng.choice(num_buckets, p=curr_probs)
    #         range_a, range_b = bucket_map[random_bucket]
    #         random_a = rng.integers(range_a[0], range_a[1])
    #         random_b = rng.integers(range_b[0], range_b[1])
    #         inferred_a[i, j] = random_a
    #         inferred_b[i, j] = random_b
    
    
    # FASTER: do all the prob choicing in one go per mask val (per distribution)
    # maps mask val to number of times mask val appears so we can vectorize np.choice
    curr_mask_counts = {}
    for i in range(256):
        for j in range(256):
            mask_val = mask_np[i, j]
            if mask_val not in curr_mask_counts:
                curr_mask_counts[mask_val] = 1
            else:
                curr_mask_counts[mask_val] += 1
    

    # maps mask_val to list of random bucks (so we go through all of them)
    random_bucket_per_mask = {}

    # maps mask_val to index of the current random bucket choice (so we go through all of them)
    random_bucket_index_map = {}

    for mask_val in curr_mask_counts:
        curr_probs = mask_abval_probs[mask_val]
        random_buckets = rng.choice(num_buckets, curr_mask_counts[mask_val], p=curr_probs)
        random_bucket_per_mask[mask_val] = random_buckets
        random_bucket_index_map[mask_val] = 0

    
    for i in range(256):
        for j in range(256): 
            mask_val = mask_np[i, j]
            curr_random_bucket = random_bucket_per_mask[mask_val][random_bucket_index_map[mask_val]]
            random_bucket_index_map[mask_val] += 1
            range_a, range_b = bucket_map[curr_random_bucket]
            random_a = min(rng.random()*bucket_size + range_a[0], 127)
            random_b = min(rng.random()*bucket_size + range_b[0], 127)
            inferred_a[i, j] = random_a
            inferred_b[i, j] = random_b

    inferred_ab_full = np.asarray([[inferred_a, inferred_b]])
    np.save(save_filename(file_idx, split=split_type), inferred_ab_full)
    if file_idx % 100 == 0:
        print("current file index: ", file_idx)
        print("\tcurrent time elapsed", time.process_time() - start_time)      

if __name__ == "__main__":
    # load frequency distribution for mask vals
    mask_abval_freqs = None
    with open("ade2k_mask_distribution.npy", "rb") as f:
        mask_abval_freqs = np.load(f)
        f.close()

    # generate probability distribution for mask vals, rows sum to 1
    mask_abval_probs = mask_abval_freqs/mask_abval_freqs.sum(axis=1,keepdims=1)

    num_buckets = 100
    bucket_size = 26
    num_files = 2000
    # prepopulate the map of bucket -> ab range
    # ab range is provided so that once bucket is picked randomly from prob dist,
    # a float val is picked randomly from ab range to introduce some noise 

    bucket_map = {}
    for i in range(num_buckets):
        range_a, range_b = bucket_to_ab_range(i)
        bucket_map[i] = (range_a, range_b)

    # set random seed
    rng = np.random.default_rng(seed=0)
    out = rng.random(5) 

    start_time = time.process_time()

    Parallel(n_jobs=8)(delayed(infer_ab_npy)(i, mask_abval_probs, start_time, split_type="val") for i in range(num_files))

    # for file_idx in range(num_files):
    #     # load mask
    #     mask_np = np.asarray(get_file_mask(file_idx, split="val"))

    #     # store final inferred mask in 1 x 2 x 256 x 256 np arr        
    #     inferred_a = np.zeros((256, 256))
    #     inferred_b = np.zeros((256, 256))

    #     # super slow code, 2.9 secs per image
    #     # for i in range(256):
    #     #     for j in range(256):
    #     #         mask_val = mask_np[i, j]
    #     #         curr_probs = mask_abval_probs[mask_val]
    #     #         random_bucket = rng.choice(num_buckets, p=curr_probs)
    #     #         range_a, range_b = bucket_map[random_bucket]
    #     #         random_a = rng.integers(range_a[0], range_a[1])
    #     #         random_b = rng.integers(range_b[0], range_b[1])
    #     #         inferred_a[i, j] = random_a
    #     #         inferred_b[i, j] = random_b
        
        
    #     # FASTER: do all the prob choicing in one go per mask val (per distribution)
    #     # maps mask val to number of times mask val appears so we can vectorize np.choice
    #     curr_mask_counts = {}
    #     for i in range(256):
    #         for j in range(256):
    #             mask_val = mask_np[i, j]
    #             if mask_val not in curr_mask_counts:
    #                 curr_mask_counts[mask_val] = 1
    #             else:
    #                 curr_mask_counts[mask_val] += 1
        

    #     # maps mask_val to list of random bucks (so we go through all of them)
    #     random_bucket_per_mask = {}

    #     # maps mask_val to index of the current random bucket choice (so we go through all of them)
    #     random_bucket_index_map = {}

    #     for mask_val in curr_mask_counts:
    #         curr_probs = mask_abval_probs[mask_val]
    #         random_buckets = rng.choice(num_buckets, curr_mask_counts[mask_val], p=curr_probs)
    #         random_bucket_per_mask[mask_val] = random_buckets
    #         random_bucket_index_map[mask_val] = 0

        
    #     for i in range(256):
    #         for j in range(256): 
    #             mask_val = mask_np[i, j]
    #             curr_random_bucket = random_bucket_per_mask[mask_val][random_bucket_index_map[mask_val]]
    #             random_bucket_index_map[mask_val] += 1
    #             range_a, range_b = bucket_map[curr_random_bucket]
    #             random_a = min(rng.random()*bucket_size + range_a[0], 127)
    #             random_b = min(rng.random()*bucket_size + range_b[0], 127)
    #             inferred_a[i, j] = random_a
    #             inferred_b[i, j] = random_b

    #     inferred_ab_full = np.asarray([[inferred_a, inferred_b]])
    #     np.save(save_filename(file_idx, split="val"), inferred_ab_full)
    #     if file_idx % 100 == 0:
    #         print("current file index: ", file_idx)      
    #         print("\tcurrent time elapsed", time.process_time() - start_time)
        

