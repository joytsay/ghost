from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import pickle
import random
import os
import cv2
import tqdm
import sys
import numpy as np
sys.path.append('..')
# from utils.cap_aug import CAP_AUG
    

class FaceEmbed(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.8):
        datasets = []
        # embeds = []
        self.N = []
        self.same_prob = same_prob
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
            # with open(f'{data_path}/embed.pkl', 'rb') as f:
            #     embed = pickle.load(f)
            #     embeds.append(embed)
        self.datasets = datasets
        # self.embeds = embeds
        self.transforms_arcface = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
            
        image_path = self.datasets[idx][item]
        # name = os.path.split(image_path)[1]
        # embed = self.embeds[idx][name]
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person

    def __len__(self):
        return sum(self.N)


class FaceEmbedVGG2(TensorDataset):
    def __init__(self, data_path, same_prob=0.8, same_identity=False):

        self.same_prob = same_prob
        self.same_identity = same_identity
                
        self.images_list = glob.glob(f'{data_path}/*/*.*g')
        self.folders_list = glob.glob(f'{data_path}/*')
        
        self.folder2imgs = {}

        for folder in tqdm.tqdm(self.folders_list):
            folder_imgs = glob.glob(f'{folder}/*')
            self.folder2imgs[folder] = folder_imgs
             
        self.N = len(self.images_list)
        
        self.transforms_arcface = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
            
        image_path = self.images_list[item]

        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)
        
        if self.same_identity:
            folder_name = '/'.join(image_path.split('/')[:-1])

        if random.random() > self.same_prob:
            image_path = random.choice(self.images_list)
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            if self.same_identity:
                image_path = random.choice(self.folder2imgs[folder_name])
                Xt = cv2.imread(image_path)[:, :, ::-1]
                Xt = Image.fromarray(Xt)
            else:
                Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person

    def __len__(self):
        return self.N

def compose_occlusion(face_img, occlusions):
    h, w, c = face_img.shape
    if len(occlusions) == 0:
        return face_img
    for occlusion in occlusions:
        # scale
        scale = random.random() * 0.5 + 0.5
        # occlusion = cv2.resize(occlusion, (), fx=scale, fy=scale)
        # rotate
        R = cv2.getRotationMatrix2D((occlusion.shape[0]/2, occlusion.shape[1]/2), random.random()*180-90, scale)
        occlusion = cv2.warpAffine(occlusion, R, (occlusion.shape[1], occlusion.shape[0]))
        print(f"occlusion.shape {occlusion.shape}")
        oh, ow, _ = occlusion.shape
        oc_color = occlusion[:, :, :3]
        oc_alpha = occlusion[:, :, 3].astype(np.float) / 255.
        oc_alpha = np.expand_dims(oc_alpha, axis=2)
        tmp = np.zeros([h+oh, w+ow, c])
        tmp[oh//2:oh//2+h, ow//2:ow//2+w, :] = face_img
        cx = random.randint(int(ow / 2) + 1, int(w + ow / 2) - 1)
        cy = random.randint(int(oh / 2) + 1, int(h + oh / 2) - 1)
        stx = cx - int(ow / 2)
        sty = cy - int(oh / 2)
        tmp[sty:sty+oh, stx:stx+ow, :] = oc_color * oc_alpha + tmp[sty:sty+oh, stx:stx+ow, :] * (1-oc_alpha)
        face_img = tmp[oh//2:oh//2+h, ow//2:ow//2+w, :].astype(np.uint8)
    return face_img

class AugmentedOcclusions(TensorDataset):
    def __init__(self, face_sets, hand_sets, obj_sets, same_prob=0.5):
        self.same_prob = same_prob
        self.face_img_paths = glob.glob(f'{face_sets}/*/*.*g')
        self.hands_data = glob.glob(f'{hand_sets}/*.png')
        self.obj_data = glob.glob(f'{obj_sets}/*/*.png')
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def gen_occlusion(self):
        p = random.random()
        occlusions = []
        if p < 0.25: # no occlusion
            pass
        elif p < 0.5: # only hand
            hand_img = cv2.imread(self.hands_data[random.randint(0, len(self.hands_data)-1)], cv2.IMREAD_UNCHANGED)
            occlusions.append(hand_img)
        elif p < 0.75: # only object
            obj_img = cv2.imread(self.obj_data[random.randint(0, len(self.obj_data)-1)], cv2.IMREAD_UNCHANGED)
            occlusions.append(obj_img)
        else: # both
            hand_img = cv2.imread(self.hands_data[random.randint(0, len(self.hands_data)-1)], cv2.IMREAD_UNCHANGED)
            occlusions.append(hand_img)
            obj_img = cv2.imread(self.obj_data[random.randint(0, len(self.obj_data)-1)], cv2.IMREAD_UNCHANGED)
            occlusions.append(obj_img)
        return occlusions

    def __getitem__(self, item):
        face_path = self.face_img_paths[item]
        face_img = cv2.imread(face_path)[:, :, ::-1]

        Xs = face_img
        p = random.random()
        if p > self.same_prob:
            Xt_path = self.face_img_paths[random.randint(0, len(self.face_img_paths)-1)]
            Xt = cv2.imread(Xt_path)[:, :, ::-1]
            Xt = compose_occlusion(Xt, self.gen_occlusion())
            same_person = 0
        else:
            Xt = compose_occlusion(face_img, self.gen_occlusion())
            same_person = 1
        return self.transforms_base(Image.fromarray(Xs)), self.transforms_base(Image.fromarray(Xt)), same_person

    def __len__(self):
        return len(self.face_img_paths)