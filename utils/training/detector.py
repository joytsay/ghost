import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from AdaptiveWingLoss.utils.utils import get_preds_fromhm
from .image_processing import torch2image


transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def detect_landmarks(inputs, model_ft):
    mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(inputs.device)
    std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(inputs.device)
    inputs = (std * inputs) + mean

    outputs, boundary_channels = model_ft(inputs)    
    pred_heatmap = outputs[-1][:, :-1, :, :].cpu() 
    pred_landmarks, _ = get_preds_fromhm(pred_heatmap)
    landmarks = pred_landmarks*4.0
    left_eyes = torch.stack((landmarks[:,60,:], landmarks[:,61,:], landmarks[:,63,:], landmarks[:,64,:], landmarks[:,67,:], landmarks[:,65,:], landmarks[:,96,:]), 1)
    right_eyes = torch.stack((landmarks[:,68,:], landmarks[:,69,:], landmarks[:,71,:], landmarks[:,72,:], landmarks[:,75,:], landmarks[:,73,:], landmarks[:,97,:]), 1)
    mouth = torch.stack((landmarks[:,88,:], landmarks[:,89,:], landmarks[:,91,:], landmarks[:,92,:], landmarks[:,95,:], landmarks[:,93,:]), 1)
    pred_heatmap_left_eyes = torch.stack((pred_heatmap[:,60,:], pred_heatmap[:,61,:], pred_heatmap[:,63,:], pred_heatmap[:,64,:], pred_heatmap[:,67,:], pred_heatmap[:,65,:], pred_heatmap[:,96,:]), 1)
    pred_heatmap_right_eyes = torch.stack((pred_heatmap[:,68,:], pred_heatmap[:,69,:], pred_heatmap[:,71,:], pred_heatmap[:,72,:], pred_heatmap[:,75,:], pred_heatmap[:,73,:], pred_heatmap[:,97,:]), 1)
    pred_heatmap_mouth = torch.stack((pred_heatmap[:,88,:], pred_heatmap[:,89,:], pred_heatmap[:,91,:], pred_heatmap[:,92,:], pred_heatmap[:,95,:], pred_heatmap[:,93,:]), 1)
    return left_eyes, right_eyes, mouth, pred_heatmap_left_eyes, pred_heatmap_right_eyes, pred_heatmap_mouth


def paint_eyes(images, left_eyes, right_eyes, mouth):
    list_eyes_mouth = []
    for i in range(len(images)):
        mask = torch2image(images[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        for j in range(left_eyes.shape[1]):
            cv2.circle(mask, (int(left_eyes[i][j][0]),int(left_eyes[i][j][1])), radius=3, color=(0,255,255), thickness=-1)
        for j in range(right_eyes.shape[1]):
            cv2.circle(mask, (int(right_eyes[i][j][0]),int(right_eyes[i][j][1])), radius=3, color=(255,0,255), thickness=-1)
        for j in range(mouth.shape[1]):
            cv2.circle(mask, (int(mouth[i][j][0]),int(mouth[i][j][1])), radius=3, color=(255,255,0), thickness=-1)
        
        mask = mask[:, :, ::-1]
        mask = transforms_base(Image.fromarray(mask))
        list_eyes_mouth.append(mask)
    tensor_eyes_mouth = torch.stack(list_eyes_mouth)
    return tensor_eyes_mouth