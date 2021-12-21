from src.detector import detect_faces
from src.utils import show_bboxes
from PIL import Image
import numpy as np
import os
import json
import argparse

def main(args):

	im_names = os.listdir(args.input_path)
	dots = {}
	for im_name in im_names:
		if im_name.split('.')[-1] == 'png' or im_name.split('.')[-1] == 'jpg' or im_name.split('.')[-1] == 'jpeg':
			name, box, landmark = image_process_squad(os.path.join(args.input_path, im_name), args.output_path)
			#f.write(f'{name};{box[0]},{box[1]},{box[2]},{box[3]}; {landmark}' + '\n')
			dots[str(name)] = {'landmarks': landmark.tolist()}
	with open(os.path.join(args.output_path, 'data.json'), 'w') as f:
		json.dump(dots, f)


def image_process_squad(path, output_folder):

    image = Image.open(path)
    bounding_boxes, landmarks = detect_faces(image)
    # Наибольшая степень уверенности в первом индексе.
    box = bounding_boxes[0] 
    landmark = landmarks[0]
    width = int(box[2] - box[0])
    height = int(box[3] - box[1])
    length = max(width, height) # Ожидаем квадрат.
    
    margin = 1.8 # Масштабируем в n раз, что бы гарантированно захватить лицо и волосы.
    length *= margin
    
    # 1. Увеличием текущий бокс до квадрата с заданными размерами.
    w_add = (length - width) // 2
    h_add = (length - height) // 2
    box[0] -= w_add
    box[2] += w_add
    box[1] -= h_add
    box[3] += h_add
    
    # Если выходим за границы изображения, сжимаем бокс пока не прекратим.
    bad_border = box[0] < 0 or box[2] > image.width or box[1] < 0 or box[3] > image.height
    while bad_border:
        squez = 2
        box[0] += squez
        box[2] -= squez
        box[1] += squez
        box[3] -= squez
        bad_border = box[0] < 0 or box[2] > image.width or box[1] < 0 or box[3] > image.height
    box = box.astype(int)
    
    img_name = path.split('\\')[-1].split('.')[-2] + '.png'
    cropped_img = image.crop(list(box[0:4]))
    cropped_img.save(os.path.join(output_folder, img_name))
    
    # Возвращается бокс от исходного изображения и лендмарки от вырезанного.
    # Поскольку бокс для вырезанного не важен.
    # А лендмарки нужны для вырезанного.
    _ , landmarks = detect_faces(cropped_img)
    
    return img_name, list(box[0:4]), landmarks


if __name__ == "__main__":
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', type=str, help='path of input images')
    parser.add_argument('-o', '--output-path', type=str, help='path of output images and info file')
    args = parser.parse_args()
    
    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()

    main(args)
