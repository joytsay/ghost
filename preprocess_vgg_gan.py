import os
import sys
import cv2
import numpy as np
import argparse
from insightface_func.face_detect_crop_single import Face_detect_crop
from pathlib import Path
from tqdm import tqdm
from torchface.models.face_alignment import FANPredictor
from torchface.models.super_resolution import CodeFormerDD
from torchface.utils.preprocessing import get_pggan_alignment
from torchface.utils.transform import antialias_warp_affine, antialias_resize

def main(args):
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    crop_size = 224
    device = "cuda"
    lms_predictor = FANPredictor(device)
    face_enhancer = CodeFormerDD(device)
    dirs = os.listdir(args.path_to_dataset)
    for i in tqdm(range(len(dirs))):
        d = os.path.join(args.path_to_dataset, dirs[i])
        dir_to_save = os.path.join(args.save_path, dirs[i])
        Path(dir_to_save).mkdir(parents=True, exist_ok=True)
        image_names = os.listdir(d)
        for image_name in image_names:
            try:
                image_path = os.path.join(d, image_name)
                image = cv2.imread(image_path)
                image = antialias_resize(image, [crop_size, crop_size])
                input_size = face_enhancer.input_size
                H, W, _ = image.shape
                results = image.copy()
                mask = np.ones((input_size, input_size, 3))
                mask = cv2.blur(mask, (5, 5))
                face_bboxs = [[0,0,image.shape[1], image.shape[0]]]
                lms = lms_predictor.predict(image, face_bboxs)[0]
                m = get_pggan_alignment(lms, output_size=face_enhancer.input_size)
                m_inv = np.linalg.inv(m)
                face_region = antialias_warp_affine(image, m[:2], (input_size, input_size),
                                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))     # gray
                enhanced_region = face_enhancer.process(face_region)
                # we need to use replicate border mode to prevent unexpected alias in the final results
                reversed_enhanced = antialias_warp_affine(enhanced_region, m_inv, (W, H), borderMode=cv2.BORDER_REPLICATE)
                reversed_mask = antialias_warp_affine(mask, m_inv[:2], (W, H))

                results = reversed_enhanced * reversed_mask + results * (1 - reversed_mask)

                results = np.clip(results, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(dir_to_save, image_name), results)
            except:
                pass
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_dataset', default='./VggFace2/VGG-Face2/data/preprocess_train', type=str)
    parser.add_argument('--save_path', default='./VggFace2-crop', type=str)
    args = parser.parse_args()
    
    main(args)
