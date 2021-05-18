
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=False).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

val_dataset = ADE2kDataset("/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/images/validation", \
                            "/home/ec2-user/colorization819/colorization/data/ADEChallengeData2016/annotations/validation", \
                            "val")

inputs, labels, inferred_ab = val_dataset.__getitem__(0)

use_ade2k = True
epoch_path = "colorizers/model_weights/epoch_6.pt"
if use_ade2k:
	colorizer_siggraph17.load_state_dict(torch.load(epoch_path, map_location=torch.device('cpu')))

# print("checkpoint 1")
# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
# (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
(tens_l_orig, tens_l_rs, img_ab_rs) = preprocess_img(img, HW=(256,256), ade2k=True)
# print("img size", img.shape)
# print("tens_l_orig", list(tens_l_orig.size()))
# print("tens_l_rs", list(tens_l_rs.size()))
if(opt.use_gpu):
	print("gpu")
	tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
# print("checkpoint 2")

print("input size", list(tens_l_rs.size()))

img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
# print("checkpoint 3a")

out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
# print("checkpoint 3b")

print("my input is size", list(tens_l_rs.size()))
model_output = colorizer_siggraph17(tens_l_rs, input_B=inferred_ab).cpu()

criterion = torch.nn.MSELoss()
loss = criterion(model_output, img_ab_rs)
print("LOSS is: ", loss.item())

out_img_siggraph17 = postprocess_tens(tens_l_orig, model_output)

# print("checkpoint 3")

plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

print("checkpoint 4")

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')
plt.show()
