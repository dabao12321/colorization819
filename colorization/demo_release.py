
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
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

use_ade2k = True
epoch_path = "colorizers/model_weights/epoch_9.pt"
if use_ade2k:
	colorizer_siggraph17.load_state_dict(torch.load(epoch_path))

# print("checkpoint 1")
# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
<<<<<<< HEAD
# (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
(tens_l_orig, tens_l_rs, img_ab_rs) = preprocess_img(img, HW=(256,256), ade2k=True)
=======
(tens_l_orig, tens_l_rs, tens_ab_rs) = preprocess_img(img, HW=(256,256), ade2k=True)
>>>>>>> hw/class
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

<<<<<<< HEAD
print("my input is size", list(tens_l_rs.size()))
model_output = colorizer_siggraph17(tens_l_rs).cpu()

criterion = torch.nn.MSELoss()
loss = criterion(model_output, img_ab_rs)
print("LOSS is: ", loss.item())

out_img_siggraph17 = postprocess_tens(tens_l_orig, model_output)
=======
(reg_outputs, class_outputs) = colorizer_siggraph17(tens_l_rs)
# print("my input is size", list(model_output.size()))
ab_rs = tens_ab_rs[:, :, ::4, ::4]
ab_norm = 110.
ab_max = 110.
ab_quant = 10.
A = 2 * ab_max / ab_quant + 1
ab_enc = encode_ab_ind(ab_rs, ab_max, ab_quant, A)
criterionCE = nn.CrossEntropyLoss()
criterionL1 = nn.L1Loss()
class_loss = 0
if torch.cuda.is_available():
	class_loss += criterionCE(class_outputs.type(torch.cuda.FloatTensor), ab_enc[:, 0, :, :].type(torch.cuda.LongTensor))
else:
	class_loss += criterionCE(class_outputs.type(torch.FloatTensor), ab_enc[:, 0, :, :].type(torch.LongTensor))
reg_loss = 10 * torch.mean(criterionL1(reg_outputs.type(torch.cuda.FloatTensor),
												tens_ab_rs.type(torch.cuda.FloatTensor)))
loss = class_loss * 1. + reg_loss
print("classification + regression loss =", loss)

out_img_siggraph17 = postprocess_tens(tens_l_orig, reg_outputs)
>>>>>>> hw/class

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
