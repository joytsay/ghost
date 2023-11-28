import os
import torch
import sys
import cv2
import numpy as np
import argparse
from insightface_func.face_detect_crop_single import Face_detect_crop
from pathlib import Path
from tqdm import tqdm

def main(args):
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    crop_size = 224

    dirs = os.listdir(args.path_to_dataset)
    print(f"path_to_dataset {args.path_to_dataset}")
    print(f"save_path {args.save_path}")
    print(f"dist {args.dist}")
    print(f"dirs {len(dirs)}")
    if args.dist == 0:
        dirs = dirs[:len(dirs)//2]
    elif args.dist == 1:
        dirs = dirs[len(dirs)//2:]
    print(f"dist dirs {len(dirs)}")
    for i in tqdm(range(len(dirs))):
        d = os.path.join(args.path_to_dataset, dirs[i])
        dir_to_save = os.path.join(args.save_path, dirs[i])
        Path(dir_to_save).mkdir(parents=True, exist_ok=True)
        
        image_names = os.listdir(d)
        for image_name in image_names:
            try:
                image_path = os.path.join(d, image_name)
                image = cv2.imread(image_path)
                cropped_image, _ = app.get(image, crop_size)
                cv2.imwrite(os.path.join(dir_to_save, image_name), cropped_image[0])
            except:
                pass
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_dataset', default='./examples/glintest/glint_subtest/', type=str)
    parser.add_argument('--save_path', default='./examples/glintest/vgg/', type=str)
    parser.add_argument('--dist', default=-1, type=int)
    args = parser.parse_args()
    
    main(args)
