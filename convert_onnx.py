import io
import numpy as np
import torch.onnx
from network.AEI_Net import AEI_Net

model_path = "saved_models_refine-Asian-1200_70_block2/G_666k.pth"
onnx_path = "saved_models_refine-Asian-1200_70_block2/G_666k.onnx"
batch_size = 1

# main model for generation
G = AEI_Net("unet", num_blocks=2, c_id=512)
G.eval()
G.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

target = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
source = torch.randn(batch_size, 512, requires_grad=True)

torch.onnx.export(G, (target, source), onnx_path, export_params=True, opset_version=11, 
                  do_constant_folding=True, 
                  input_names = ['target', 'source_emb'], output_names = ['Y_st'])