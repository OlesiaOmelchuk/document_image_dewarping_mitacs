import torch
import torch.nn as nn

# ---------------------------
# Loss Functions
# ---------------------------
def tv_loss(flow):
    """Total variation loss for smooth flow fields"""
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    return torch.mean(dx) + torch.mean(dy)

def weak_jacobian_penalty(flow):
    """Weak Jacobian penalty to prevent fold-overs"""
    B, C, H, W = flow.shape
    # flow is already normalized to [-1,1], so gradients are consistent
    grad_x = torch.gradient(flow[:, 0], dim=2)[0]  # ∂u/∂x
    grad_y = torch.gradient(flow[:, 1], dim=1)[0]  # ∂v/∂y

    # Approximate Jacobian determinant
    jac_det = (1 + grad_x) * (1 + grad_y) - grad_x * grad_y
    return torch.mean(torch.relu(-jac_det))  # penalize negative det (folds)

class FlowLoss(nn.Module):
    def __init__(self, tv_weight=0.01, jac_weight=0.002):
        super().__init__()
        self.tv_weight = tv_weight
        self.jac_weight = jac_weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred_flow, target_flow):
        # L1 loss for flow regression
        l1_loss = self.l1_loss(pred_flow, target_flow)
        
        # Regularization
        tv_loss_val = tv_loss(pred_flow)
        jac_loss_val = weak_jacobian_penalty(pred_flow)
        
        total_loss = l1_loss + self.tv_weight * tv_loss_val + self.jac_weight * jac_loss_val
        return total_loss, l1_loss, tv_loss_val, jac_loss_val