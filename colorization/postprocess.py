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

# print("checkpoint 1")
# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs, tens_ab_rs) = preprocess_img(img, HW=(256,256), ade2k=True)
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

class_colorizer_siggraph17 = siggraph17(pretrained=False).eval()
device = torch.device('cpu')
class_colorizer_siggraph17.load_state_dict(torch.load("colorizers/model_weights_class0/epoch_9.pt", map_location=device))
# class_colorizer_siggraph17.cpu()

(reg_output, class_output) = class_colorizer_siggraph17(tens_l_rs, output=True)
print("should be something before this?")
class_output.cpu()
print("my input is size", list(class_output.size()))
ab_rs = tens_ab_rs[:, :, ::4, ::4]
ab_norm = 110.
ab_max = 110.
ab_quant = 10.
A = 2 * ab_max / ab_quant + 1
ab_enc = encode_ab_ind(ab_rs, ab_max, ab_quant, A)
loss = 0
criterion = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
# 	loss = criterion(class_output.type(torch.cuda.FloatTensor), ab_enc[:, 0, :, :].type(torch.cuda.LongTensor)).item()
# else:
loss += criterion(class_output.type(torch.FloatTensor), ab_enc[:, 0, :, :].type(torch.LongTensor)).item()
print("classification loss =", loss)

out_img_siggraph17 = postprocess_tens(tens_l_orig, reg_output)
# class_img_siggraph17 = postprocess_tens_class(tens_l_orig, class_output)

# print("checkpoint 3")

plt.imsave('%s_class_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

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

plt.subplot(2,2,4)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')
plt.show()
