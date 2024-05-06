print("started imports")

import sys
import argparse
import time
import cv2
import wandb
import os
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler as scheduler
from utils.training.image_processing import make_image_list
import dlib
# custom imports
sys.path.append('./apex/')

from apex import amp
from network.AEI_Net import *
from network.HEAR_Net import *
from utils.training.Dataset import AugmentedOcclusions
from onnx2torch import convert


print("finished imports")
L1 = torch.nn.L1Loss()

def train_one_epoch(G: 'generator model', 
                    net: 'hearnet model', 
                    opt: "hearnet_opt", 
                    netArc: 'netArc model',
                    args: 'Args Namespace',
                    dataloader: torch.utils.data.DataLoader,
                    device: 'torch device',
                    epoch:int):
    show_step = args.show_step
    save_epoch = args.save_epoch
    for iteration, data in enumerate(dataloader):
        if args.max_steps > 0 and iteration > args.max_steps:
            break
        start_time = time.time()
        
        Xs_orig, Xt_orig, Xs, Xt, same_person = data
        Xs_orig = Xs_orig.to(device)
        Xt_orig = Xt_orig.to(device)
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        with torch.no_grad():
            embed_s = netArc(F.interpolate(Xs_orig, [112, 112], mode='bilinear', align_corners=False))
            embed_t = netArc(F.interpolate(Xt_orig, [112, 112], mode='bilinear', align_corners=False))
        same_person = same_person.to(device)

        # train HEAR
        opt.zero_grad()
        with torch.no_grad():
            Yst_hat, _ = G(Xt, embed_s)
            Ytt, _ = G(Xt, embed_t)

        dYt = Xt - Ytt
        hear_input = torch.cat((Yst_hat, dYt), dim=1)
        Yst = net(hear_input)

        Yst_aligned = Yst

        id_Yst = netArc(F.interpolate(Yst_aligned, [112, 112], mode='bilinear', align_corners=True))

        L_id = (1 - torch.cosine_similarity(embed_s, id_Yst, dim=1)).mean()

        L_chg = L1(Yst_hat, Yst)

        L_rec = torch.sum(0.5 * torch.mean(torch.pow(Yst - Xt, 2).reshape(args.batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

        loss = L_id + L_chg + L_rec
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()

        # loss.backward()
        opt.step()

        batch_time = time.time() - start_time
        if iteration % show_step == 0:
            images = [Xs, Xt, Ytt, Yst_hat, Yst]
            image = make_image_list(images)
            if args.use_wandb:
                wandb.log({"gen_images":wandb.Image(image, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})
            else:
                cv2.imwrite('./images/HEAR_generated_image.jpg', image[:,:,::-1])
        print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
        print(f'loss: {loss.item()} batch_time: {batch_time}s')
        print(f'L_id: {L_id.item()} L_chg: {L_chg.item()} L_rec: {L_rec.item()}')
        if args.use_wandb:
            wandb.log({"loss": loss.item(),
                       "L_id": L_id.item(),
                       "L_chg": L_chg.item(),
                       "L_rec": L_rec.item()})
        if iteration % save_epoch == 0:
            torch.save(net.state_dict(), f'./saved_models_{args.run_name}/HEAR_latest.pth')


def train(args, device):
    # training params
    batch_size = args.batch_size
    lr = args.lr
    max_epoch = args.max_epoch
    optim_level = args.optim_level

    # initializing main models
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512).to(device)
    G.eval()

    net = HearNet()
    net.train()
    net.to(device)

    netArc = convert(args.arcface_onnx_path)
    netArc = netArc.cuda()
    netArc.eval()

    opt = optim.Adam(net.parameters(), lr=lr, betas=(0, 0.999))
    net, opt = amp.initialize(net, opt, opt_level=optim_level)
        
    if args.pretrained:
        try:
            if args.G_path:
                G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')), strict=False)
                print("Loaded pretrained weights for G")
            if args.HEAR_path:
                net.load_state_dict(torch.load(args.HEAR_path, map_location=torch.device('cpu')), strict=False)
                print("Loaded pretrained weights for HEARNET")
        except FileNotFoundError as e:
            print("Not found pretrained weights. Continue without any pretrained weights.")

    # Set up dlib face detector and predictor
    args.detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model()

    args.predictor = dlib.shape_predictor(path_to_dlib_model)

    # Extract data from code
    mask_code = "".join(args.code.split()).split(",")
    args.code_count = np.zeros(len(mask_code))
    args.mask_dict_of_dict = {}


    for i, entry in enumerate(mask_code):
        mask_dict = {}
        mask_color = ""
        mask_texture = ""
        mask_type = entry.split("-")[0]
        if len(entry.split("-")) == 2:
            mask_variation = entry.split("-")[1]
            if "#" in mask_variation:
                mask_color = mask_variation
            else:
                mask_texture = mask_variation
        mask_dict["type"] = mask_type
        mask_dict["color"] = mask_color
        mask_dict["texture"] = mask_texture
        args.mask_dict_of_dict[i] = mask_dict
    
    dataset = AugmentedOcclusions(args)
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    print(torch.backends.cudnn.benchmark)
    
    for epoch in range(0, max_epoch):
        train_one_epoch(G,
                        net,
                        opt,
                        netArc,
                        args,
                        dataloader,
                        device,
                        epoch)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('cuda is not available. using cpu. check if it\'s ok')
    
    print("Starting traing")
    train(args, device=device)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset params
    parser.add_argument('--dataset_path', default='./examples/heartest/faces', help='Path to the faces dataset.')
    parser.add_argument('--arcface_onnx_path', default=None, help='Path to source arcface emb extractor')
    parser.add_argument('--G_path', default='', help='Path to pretrained weights for G. Only used if pretrained=True')
    parser.add_argument('--HEAR_path', default='', help='Path to pretrained weights for HEARNET. Only used if pretrained=True')
    # training params you may want to change
    parser.add_argument('--backbone', default='unet', const='unet', nargs='?', choices=['unet', 'linknet', 'resnet'], help='Backbone for attribute encoder')
    parser.add_argument('--num_blocks', default=2, type=int, help='Numbers of AddBlocks at AddResblock')
    parser.add_argument('--pretrained', default=True, type=bool, help='If using the pretrained weights for training or not')
    # info about this run
    parser.add_argument('--use_wandb', default=False, type=bool, help='Use wandb to track your experiments or not')
    parser.add_argument('--run_name', required=True, type=str, help='Name of this run. Used to create folders where to save the weights.')
    parser.add_argument('--wandb_project', default='your-project-name', type=str)
    parser.add_argument('--wandb_entity', default='your-login', type=str)
    # training params you probably don't want to change
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--same_prob', default=0.25, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--show_step', default=10, type=int)
    parser.add_argument('--save_epoch', default=1000, type=int)
    parser.add_argument('--optim_level', default='O2', type=str)
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to either the folder containing images or the image itself",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="glasses",
        choices=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "mic", "hand", "glasses", "random", "all"],
        help="Type of the mask to be applied. Available options: all, surgical_blue, surgical_green, N95, cloth",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Type of the pattern. Available options in masks/textures",
    )

    parser.add_argument(
        "--pattern_weight",
        type=float,
        default=0.5,
        help="Weight of the pattern. Must be between 0 and 1",
    )

    parser.add_argument(
        "--color",
        type=str,
        default="#0473e2",
        help="Hex color value that need to be overlayed to the mask",
    )

    parser.add_argument(
        "--color_weight",
        type=float,
        default=0.5,
        help="Weight of the color intensity. Must be between 0 and 1",
    )

    parser.add_argument(
        "--code",
        type=str,
        # default="cloth-masks/textures/check/check_4.jpg, cloth-#e54294, cloth-#ff0000, cloth, cloth-masks/textures/others/heart_1.png, cloth-masks/textures/fruits/pineapple.png, N95, surgical_blue, surgical_green",
        default="",
        help="Generate specific formats",
    )


    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Turn verbosity on"
    )
    parser.add_argument(
        "--write_original_image",
        dest="write_original_image",
        action="store_true",
        help="If true, original image is also stored in the masked folder",
    )
    args = parser.parse_args()
    
    if args.use_wandb==True:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, settings=wandb.Settings(start_method='fork'))
        config = wandb.config
        config.dataset_path = args.dataset_path
        config.pretrained = args.pretrained
        config.run_name = args.run_name
        config.G_path = args.G_path
        config.batch_size = args.batch_size
        config.lr = args.lr
        config.max_epoch = args.max_epoch
    elif not os.path.exists('./images'):
        os.mkdir('./images')
    
    if not os.path.exists(f'./saved_models_{args.run_name}'):
        os.mkdir(f'./saved_models_{args.run_name}')
    
    main(args)
