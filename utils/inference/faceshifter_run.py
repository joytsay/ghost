import torch
import numpy as np


def faceshifter_batch(source_emb: torch.tensor, 
                      target: torch.tensor,
                      G, use_onnx=False) -> np.ndarray:
    """
    Apply faceshifter model for batch of target images
    """
    
    bs = target.shape[0]
    assert target.ndim == 4, "target should have 4 dimentions -- B x C x H x W"
    
    if bs > 1:
        source_emb = torch.cat([source_emb]*bs)
    with torch.no_grad():
        if use_onnx:
            print(f"faceshifter_batch use_onnx")
            ort_inputs = {'Xt': target.detach().cpu().numpy(), 'z_id': source_emb.detach().cpu().numpy()}
            ort_outs = G.run(None, ort_inputs)
            Y_st = torch.Tensor(ort_outs[0]).to("cuda")
        else:
            print(f"faceshifter_batch use_pth")
            Y_st, _ = G(target, source_emb)
        Y_st = (Y_st.permute(0, 2, 3, 1)*0.5 + 0.5)*255
        Y_st = Y_st[:, :, :, [2,1,0]].type(torch.uint8)
        Y_st = Y_st.cpu().detach().numpy()    
    return Y_st