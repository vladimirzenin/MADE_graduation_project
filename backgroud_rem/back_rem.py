import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet

# weight of model
ckpt_path = "modnet_photographic_portrait_matting.ckpt"

# define hyper-parameters
# ref_size = 512 -> -q parameter

def main(args):
	
	ref_size = args.quality
	
	# define image to tensor transform
	im_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]
	)

	# create MODNet and load the pre-trained ckpt
	modnet = MODNet(backbone_pretrained=False)
	modnet = nn.DataParallel(modnet).cuda()
	modnet.load_state_dict(torch.load(ckpt_path))
	modnet.eval()

	# inference images
	im_names = os.listdir(args.input_path)
	for im_name in im_names:
		if im_name.split('.')[-1] == 'png' or im_name.split('.')[-1] == 'jpg' or im_name.split('.')[-1] == 'jpeg':
			# read image
			print(im_name)
			im = Image.open(os.path.join(args.input_path, im_name)).convert('RGB')

			# unify image channels to 3
			im = np.asarray(im)

			im_base = im.copy()

			if len(im.shape) == 2:
				im = im[:, :, None]
			if im.shape[2] == 1:
				im = np.repeat(im, 3, axis=2)
			elif im.shape[2] == 4:
				im = im[:, :, 0:3]

			# convert image to PyTorch tensor
			im = Image.fromarray(im)
			im = im_transform(im)

			# add mini-batch dim
			im = im[None, :, :, :]

			# resize image for input
			im_b, im_c, im_h, im_w = im.shape
			if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
				if im_w >= im_h:
					im_rh = ref_size
					im_rw = int(im_w / im_h * ref_size)
				elif im_w < im_h:
					im_rw = ref_size
					im_rh = int(im_h / im_w * ref_size)
			else:
				im_rh = im_h
				im_rw = im_w

			im_rw = im_rw - im_rw % 32
			im_rh = im_rh - im_rh % 32
			im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

			# inference
			_, _, matte = modnet(im.cuda(), True)

			# resize and save matte
			matte_int = F.interpolate(matte, size=(im_h, im_w), mode='area')
			matte_np = matte_int[0][0].data.cpu().numpy()

			# save result
			ones = np.ones((im_h, im_w, 1))
			base_expand = np.concatenate((im_base, ones * 255), axis=2)

			ones = np.ones((im_h, im_w))
			mask = torch.tensor([ones, ones, ones, matte_np]).permute(1,2,0).numpy()

			img_name = 'out_' + im_name.split('.')[0] + '.png'
			Image.fromarray(((base_expand * mask).astype('uint8'))).save(os.path.join(args.output_path, img_name))


if __name__ == "__main__":
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', type=str, help='path of input images')
    parser.add_argument('-o', '--output-path', type=str, help='path of output images')
    parser.add_argument('-q', '--quality', type=int, default=512, help='size of input image before forward in network')
    args = parser.parse_args()
    
    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()
    
    main(args)
