import torch
from torch import nn
import numpy as np

# 3D global normalized cross correlation
class GlobalNCC2D(nn.Module):
    def __init__(self, eps=1e-5):
        super(GlobalNCC2D, self).__init__()
        self.eps = eps

    def forward(self, I, J):
        assert I.dim() == 4     # [BS, CH, D, H, W]
        assert I.size() == J.size()

        # compute global means
        I_mean = I.mean(dim=(2, 3), keepdim=False)
        J_mean = J.mean(dim=(2, 3), keepdim=False)
        I2_mean = (I * I).mean(dim=(2, 3), keepdim=False)
        J2_mean = (J * J).mean(dim=(2, 3), keepdim=False)
        IJ_mean = (I * J).mean(dim=(2, 3), keepdim=False)

        # compute global NCC
        CC_mean = IJ_mean - I_mean * J_mean
        I_var_mean = I2_mean - I_mean * I_mean
        J_var_mean = J2_mean - J_mean * J_mean

        NCC2 = CC_mean * CC_mean / (I_var_mean * J_var_mean + self.eps)

        return NCC2.mean()

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def zecon_loss(model, pred, real, ref=None, t=None, z=None, non_diff=False):
    if non_diff:
        loss = zecon_loss_direct(model, pred, real, non_diff=non_diff)
    else:
        loss = zecon_loss_direct(model, pred, real, ref, torch.zeros_like(t), torch.zeros_like(z), non_diff=non_diff)
    return loss.mean()


def zecon_loss_direct(Unet, x_in, y_in, ref=None, t=None, z=None, non_diff=False):
    total_loss = 0
    nce_layers = [0,2,5,8,11]
    if non_diff:
        nce_layers = [1, 4, 7, 10, 13]
    num_patches=256

    l2norm = Normalize(2)
    if non_diff:
        feat_q = Unet.module.module.forward_enc(x_in, nce_layers)
        feat_k = Unet.module.module.forward_enc(y_in, nce_layers)
    else:
        feat_q = Unet.module.forward_enc(torch.cat((x_in, ref), axis=1) ,t, z, nce_layers)
        feat_k = Unet.module.forward_enc(torch.cat((y_in, ref), axis=1) ,t, z, nce_layers)
    patch_ids = []
    feat_k_pool = []
    feat_q_pool = []
    
    for feat_id, feat in enumerate(feat_k):
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)       # [B,ch,h,w] > [B,h*w,ch]

        patch_id = np.random.permutation(feat_reshape.shape[1])
        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
        
        patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
        x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        
        patch_ids.append(patch_id)
        x_sample = l2norm(x_sample)
        feat_k_pool.append(x_sample)
    
    for feat_id, feat in enumerate(feat_q):
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)       # [B,ch,h,w] > [B,h*w,ch]

        patch_id = patch_ids[feat_id]

        patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
        x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        
        x_sample = l2norm(x_sample)
        feat_q_pool.append(x_sample)

    for f_q, f_k in zip(feat_q_pool, feat_k_pool):
        loss = PatchNCELoss(f_q, f_k)
        total_loss += loss.mean()
    return total_loss.mean()

def PatchNCELoss(feat_q, feat_k, batch_size=1, nce_T = 0.07):
    # feat_q : n_patch x 512
    # feat_q : n_patch x 512
    batch_size = batch_size
    nce_T = nce_T
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    mask_dtype = torch.bool

    num_patches = feat_q.shape[0]
    dim = feat_q.shape[1]
    feat_k = feat_k.detach()
    
    # pos logit 
    l_pos = torch.bmm(
        feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
    l_pos = l_pos.view(num_patches, 1)

    # reshape features to batch size
    feat_q = feat_q.view(batch_size, -1, dim)
    feat_k = feat_k.view(batch_size, -1, dim)
    npatches = feat_q.size(1)
    l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

    # diagonal entries are similarity between same features, and hence meaningless.
    # just fill the diagonal with very small number, which is exp(-10) and almost zero
    diagonal = torch.eye(npatches, device=feat_q.device, dtype=mask_dtype)[None, :, :]
    l_neg_curbatch.masked_fill_(diagonal, -10.0)
    l_neg = l_neg_curbatch.view(-1, npatches)

    out = torch.cat((l_pos, l_neg), dim=1) / nce_T

    loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                    device=feat_q.device))

    return loss