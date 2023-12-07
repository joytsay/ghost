import os
import sys
import cv2
import numpy as np
import argparse
from insightface_func.face_detect_crop_single import Face_detect_crop
from pathlib import Path
from tqdm import tqdm
from torchface.models.face_alignment import FANPredictor
from torchface.models.face_detection import SCRFDDetector
from torchface.models.super_resolution import CodeFormerDD
from torchface.utils.preprocessing import get_pggan_alignment_from_template
from torchface.utils.transform import antialias_warp_affine, antialias_resize
from insightface.utils import face_align

def extract_5p(lm):
    lm5p = np.stack([
        np.mean(lm[[36, 39], :], 0),  # left_eye center (mean of left_eye corners)
        np.mean(lm[[42, 45], :], 0),  # right_eye (mean of right_eye corners)
        lm[30, :],  # nose
        lm[48, :],  # left mouth corner
        lm[54, :]  # right mouth corner
        ])
    return lm5p

def main(args):
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    crop_size = 224
    device = "cuda"
    lms_predictor = FANPredictor(device)
    face_detector = SCRFDDetector(device)
    face_enhancer = CodeFormerDD(device)
    dirs = os.listdir(args.path_to_dataset)
    print(f"path_to_dataset {args.path_to_dataset}")
    print(f"gan_path {args.gan_path}")
    print(f"save_path {args.save_path}")
    print(f"dist {args.dist}")
    print(f"dirs {len(dirs)}")
    if args.dist == 0:
        dirs = dirs[:len(dirs)//4]
    elif args.dist == 1:
        dirs = dirs[len(dirs)//4:len(dirs)//2]
    elif args.dist == 2:
        dirs = dirs[len(dirs)//2:-len(dirs)//4]
    elif args.dist == 3:
        dirs = dirs[-len(dirs)//4:]
    print(f"dist dirs {len(dirs)}")
    for i in tqdm(range(len(dirs))):
        d = os.path.join(args.path_to_dataset, dirs[i])
        dir_to_save = os.path.join(args.save_path, dirs[i])
        dir_to_gan = os.path.join(args.gan_path, dirs[i])
        Path(dir_to_save).mkdir(parents=True, exist_ok=True)
        Path(dir_to_gan).mkdir(parents=True, exist_ok=True)
        image_names = os.listdir(d)
        for image_name in image_names:
            try:
                input_size = face_enhancer.input_size
                image_path = os.path.join(d, image_name)
                image = cv2.imread(image_path)
                image = antialias_resize(image, [input_size, input_size])
                value = 0
                pad = int(0.25*input_size)
                cval = np.array([[value, value], [value, value], [0, 0]], dtype=object)  # Ragged.
                image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), constant_values=cval)
                H, W, _ = image.shape
                results = image.copy()
                face_bboxs = face_detector.predict(image)
                if len(face_bboxs) == 0:
                    continue
                mask = np.ones((input_size, input_size, 3))
                mask = cv2.blur(mask, (5, 5))
                lms = lms_predictor.predict(image, face_bboxs)[0]
                m = get_pggan_alignment_from_template(lms, input_size)
                m_inv = np.linalg.inv(m)
                face_region = antialias_warp_affine(image, m[:2], (input_size, input_size),
                                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))     # gray
                enhanced_region = face_enhancer.process(face_region[..., ::-1])
                # we need to use replicate border mode to prevent unexpected alias in the final results
                reversed_enhanced = antialias_warp_affine(enhanced_region[..., ::-1], m_inv, (W, H), borderMode=cv2.BORDER_REPLICATE)
                reversed_mask = antialias_warp_affine(mask, m_inv[:2], (W, H))
                results = reversed_enhanced * reversed_mask + results * (1 - reversed_mask)
                results = np.clip(results, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(dir_to_gan, image_name), results)
                cropped_image, _ = face_align.norm_crop2(results, extract_5p(lms), crop_size)
                cv2.imwrite(os.path.join(dir_to_save, image_name), cropped_image)
            except:
                pass
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_dataset', default='./examples/glintest/glint_subtest/', type=str)
    parser.add_argument('--gan_path', default='./examples/glintest/gan_cf/', type=str)
    parser.add_argument('--save_path', default='./examples/glintest/vgg/', type=str)
    parser.add_argument('--dist', default=-1, type=int)
    args = parser.parse_args()
    
    main(args)
