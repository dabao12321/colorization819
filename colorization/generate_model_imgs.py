
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from joblib import Parallel, delayed
import os

from colorizers import *

def generate_img(model, dataset, index, file_dir="model_name_imgs/", replace=False):
    ade2k_dir = "data/ADEChallengeData2016/images/validation/"
    file_prefix = "ADE_val_"

    index += 1
    num_zeros = 8 - len(str(index))
    file_prefix += "0"*num_zeros + str(index)

    input_img_path = ade2k_dir + file_prefix + ".jpg"

    output_img_path = file_dir + file_prefix + ".png"
    if not replace and os.path.exists(output_img_path):
        return

    inputs, labels = dataset.__getitem__(index - 1, inferred_ab=False)


    img = load_img(input_img_path)
    (tens_l_orig, tens_l_rs, img_ab_rs) = preprocess_img(img, HW=(256,256), ade2k=True)

    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    # print("checkpoint 3a")

    # edit this!
    model_output = colorizer_siggraph17(tens_l_rs).cpu()
    # criterion = torch.nn.MSELoss()
    # loss = criterion(model_output, img_ab_rs)

    out_img_siggraph17 = postprocess_tens(tens_l_orig, model_output)

    plt.imsave(output_img_path, out_img_siggraph17)


if __name__ == "__main__":

    # load colorizers
    colorizer_siggraph17 = siggraph17(pretrained=False).eval()

    val_dataset = ADE2kDataset("/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/images/validation", \
                                "/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/annotations/validation", \
                                "val")

    use_ade2k = True
    epoch_path = "colorizers/model_weights_nopretrain/epoch_9.pt"
    if use_ade2k:
        colorizer_siggraph17.load_state_dict(torch.load(epoch_path, map_location=torch.device('cpu')))


    # generate_img(colorizer_siggraph17, val_dataset, 0, file_dir="model_output_imgs/")

    # randomly generated 102 image indices
    sample_ids = [0, 2, 3, 5, 1441, 807, 796, 1771, 112, \
        1727, 924, 451, 1991, 755, 1403, 1468, 1056, 334, 1160, \
        756, 1713, 1430, 418, 1127, 636, 1449, 644, 1402, 1219, \
        891, 363, 843, 189, 1194, 976, 557, 603, 1596, 9, 1157, \
        884, 1303, 1547, 1811, 833, 1981, 888, 687, 1371, 1390, \
        1898, 791, 121, 124, 716, 589, 727, 839, 1759, 540, 1650, \
        1876, 554, 479, 500, 1797, 623, 1987, 1504, 1488, 764, 129, \
        1968, 1530, 39, 847, 979, 1786, 852, 1594, 616, 1083, \
        1730, 1664, 1267, 1644, 275, 338, 331, 294, 1272, 871, 607, \
        26, 1165, 1834, 1058, 1985, 302, 139]
    
    sample_ids.sort()
    
    print(len(sample_ids))
    print(len(set(sample_ids)))

    Parallel(n_jobs=4)(delayed(generate_img)(colorizer_siggraph17, val_dataset, i, file_dir="baseline_model_imgs/") for i in sample_ids)

    # inputs, labels, masks = val_dataset.__getitem__(2)

    # # print("checkpoint 1")
    # # default size to process images is 256x256
    # # grab L channel in both original ("orig") and resized ("rs") resolutions
    # img = load_img(opt.img_path)
    # # (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    # (tens_l_orig, tens_l_rs, img_ab_rs) = preprocess_img(img, HW=(256,256), ade2k=True)
    # # print("img size", img.shape)
    # # print("tens_l_orig", list(tens_l_orig.size()))
    # # print("tens_l_rs", list(tens_l_rs.size()))
    # if(opt.use_gpu):
    #     print("gpu")
    #     tens_l_rs = tens_l_rs.cuda()

    # # colorizer outputs 256x256 ab map
    # # resize and concatenate to original L channel
    # # print("checkpoint 2")

    # img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    # # print("checkpoint 3a")

    # model_output = colorizer_siggraph17(tens_l_rs, mask_B=masks).cpu()
    # criterion = torch.nn.MSELoss()
    # loss = criterion(model_output, img_ab_rs)
    # print("LOSS is: ", loss.item())

    # out_img_siggraph17 = postprocess_tens(tens_l_orig, model_output)

    # # print("checkpoint 3")

    # plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

    # print("checkpoint 4")
