import cv2
import os
import importlib
import numpy as np
from glob import glob 

import torch
from torchvision.transforms import ToTensor

from utils.option import args
#import argparse

def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image


def get_face_hair_mask(img_mask):
    im_msk_np = np.asarray(img_mask)
    im_2 = im_msk_np.copy()
    im_11 = im_msk_np.copy()
    im_13 = im_msk_np.copy()
    im_1 = im_msk_np.copy()
    
    # 1 - головной убор -> 38 при чтении cv2
    # 2 - волосы -> 75 при чтении cv2
    # 11 - лицо и шея (atr mask)
    # 13 - лицо (lip mask) -> 72 при чтении cv2
    im_2[im_msk_np != 75] = 0
    im_11[im_msk_np != 11] = 0
    im_13[im_msk_np != 72] = 0
    im_1[im_msk_np != 38] = 0
    im_msk_np = im_2 + im_11 + im_13 + im_1

    im_msk_np[im_msk_np == 75] = 255
    im_msk_np[im_msk_np == 11] = 255
    im_msk_np[im_msk_np == 72] = 255
    im_msk_np[im_msk_np == 38] = 255
    
    return im_msk_np


def main(args):

    input_path = args.dir_image
    input_mask_path = args.dir_mask
    output_path = args.outputs
    type = 'i' #args.type
    # prepare dataset
    image_paths = []
    for ext in ['.jpg', '.png']: 
        image_paths.extend(glob(os.path.join(input_path, '*'+ext)))
    image_paths.sort()
    mask_paths=[]
    for ext in ['.jpg', '.png']: 
        mask_paths.extend(glob(os.path.join(input_mask_path, '*'+ext)))
    mask_paths.sort()
    
    os.makedirs(output_path, exist_ok=True)
    
    if type == 'i':
        # interior inpaint
        pre_train = 'pretrains/interior_inp.pt'
    elif type == 'p':
        # person inpaint
        pre_train = 'pretrains/person_inp.pt'
    
    # Model and version
    net = importlib.import_module('model.aotgan')
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load(pre_train, map_location='cpu'))
    model.eval()

    for ipath, mpath in zip(image_paths, mask_paths):
        split_fn = os.path.basename(ipath).split('.')
        filename = split_fn[0]
        
        image = cv2.imread(ipath, cv2.IMREAD_COLOR)
        h_img, w_img, channels = image.shape
        orig_img = cv2.resize(image, (512, 512))
        
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        mask = get_face_hair_mask(mask)
        msk_orig_img = cv2.resize(mask, (512, 512))
        
        img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
        h, w, c = orig_img.shape
        mask = np.asarray(msk_orig_img, np.uint8)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
        image_copy = orig_img.copy()
        
        with torch.no_grad():
        	mask_tensor = (ToTensor()(mask)).unsqueeze(0)
        	masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
        	pred_tensor = model(masked_tensor, mask_tensor)
        	comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))

        	pred_np = postprocess(pred_tensor[0])
        	masked_np = postprocess(masked_tensor[0])
        	comp_np = postprocess(comp_tensor[0])
        
        if w_img > h_img:
            aspect_ratio = float(w_img)/h_img
            comp_np = cv2.resize(comp_np, (int(512 * aspect_ratio), 512), cv2.INTER_AREA)
            masked_np = cv2.resize(masked_np, (int(512 * aspect_ratio), 512), cv2.INTER_AREA)
        elif h_img > w_img:
            aspect_ratio = float(h_img)/w_img
            comp_np = cv2.resize(comp_np, (512, int(512 * aspect_ratio)), cv2.INTER_AREA)
            masked_np = cv2.resize(masked_np, (512, int(512 * aspect_ratio)), cv2.INTER_AREA)
        
        #cv2.imwrite(os.path.join(output_path, f'{filename}_pred.png'), pred_np)
        cv2.imwrite(os.path.join(output_path, f'{filename}_comp.png'), comp_np)
        #cv2.imwrite(os.path.join(output_path, f'{filename}_mask.png'), masked_np)
        print('[**] save successfully!')


if __name__ == '__main__':
    main(args)
