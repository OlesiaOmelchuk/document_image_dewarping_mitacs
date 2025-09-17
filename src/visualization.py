import torch
import torch.nn.functional as F
import cv2
import numpy as np
import wandb  # Optional for logging
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


# --- Visualization helper ---
def prepare_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        if tensor.ndim == 4:
            tensor = tensor[0]  # remove batch dimension
        if tensor.shape[0] in [1, 3, 4]:  # channels first
            tensor = tensor.transpose(1, 2, 0)  # convert to channels last
        return tensor
    if tensor.requires_grad:
        tensor = tensor.detach()
    tensor = tensor.cpu().numpy()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] in [1, 3, 4]:
        tensor = tensor.transpose(1, 2, 0)
    return tensor

def visualize_flow(flow, save_path=None, verbose=True):
    """Visualize flow field as RGB image"""
    try:
        flow = flow.squeeze(0).cpu().numpy()
    except:
        flow = flow.squeeze(0)
    
    # Convert flow to HSV color representation
    h, w = flow.shape[1:]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Magnitude and angle
    mag, ang = cv2.cartToPolar(flow[0], flow[1])
    
    # Normalize for visualization
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to BGR and save
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    if save_path is not None:
        cv2.imwrite(save_path, bgr)

    # Create plot
    if verbose:
        plt.figure(figsize=(10, 8))
        plt.imshow(bgr)
        plt.axis('off')
        plt.title('Optical Flow Visualization')
        plt.tight_layout()
        plt.show()
        
    return bgr

def apply_bm_doc3d(img, bm_pix, align_corners=True, padding_mode="border", verbose=False, save_path=None):
    """
    Warp an image using a backward map in pixel coordinates.

    Args:
        img: (B, C, H, W) tensor in [0,1], warped image
        bm_pix: (B, 2, H, W) tensor in pixels, backward map (absolute coords)
                bm_pix[:,0] = x pixel coords
                bm_pix[:,1] = y pixel coords
        align_corners: bool, matches normalization convention in grid_sample
        padding_mode: str, 'border' or 'zeros'

    Returns:
        rectified: (B, C, H, W) tensor, unwarped image
    """
    # if len(img.shape) == 3:
    #     img = img.unsqueeze(0)
    #     bm_pix = bm_pix.unsqueeze(0)
        
    B, C, H, W = img.shape

    # convert pixel coords -> normalized [-1,1]
    if align_corners:
        norm_x = (bm_pix[:, 0, :, :] / (W - 1)) * 2 - 1
        norm_y = (bm_pix[:, 1, :, :] / (H - 1)) * 2 - 1
    else:
        norm_x = (2 * bm_pix[:, 0, :, :] + 1) / W - 1
        norm_y = (2 * bm_pix[:, 1, :, :] + 1) / H - 1

    grid = torch.stack([norm_x, norm_y], dim=-1)  # (B,H,W,2)

    rectified = F.grid_sample(
        img, grid, mode="bilinear",
        padding_mode=padding_mode, align_corners=align_corners
    )

    # For PyTorch tensors with requires_grad
    def prepare_tensor(tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()
        tensor = tensor.cpu().numpy()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.shape[0] in [1, 3, 4]:
            tensor = tensor.transpose(1, 2, 0)
        return tensor
        
    if verbose:
        img_display = prepare_tensor(img)
        rectified_display = prepare_tensor(rectified)
        
        f,axrr=plt.subplots(1,2)
        for ax in axrr:
            ax.set_xticks([])
            ax.set_yticks([])
        axrr[0].imshow(img_display)
        axrr[0].title.set_text('input')
        axrr[1].imshow(rectified_display)
        axrr[1].title.set_text('unwarped')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        
    return rectified

def visualize_epoch_results(model, dataloader, device, epoch, max_batches=1):
    model.eval()
    images, target_flows, target_flows_norm = next(iter(dataloader))
    print(images.shape)
    # image_tensor, target_flow_tensor, target_flow_norm_tensor = dataloader[0]

    # image_tensor, bm_tensor, _ = dataset[i]  # image: (3,H,W), bm: (2,H,W)
    image_tensor = images[0].unsqueeze(0).to(device)   # -> (1,3,H,W)
    target_flow_tensor    = target_flows[0].unsqueeze(0).to(device)      # -> (1,2,H,W)

    # images = images.to(device)
    # target_flows = target_flows.to(device)

    with torch.no_grad():
        pred_flows = model(image_tensor)

    # B, _, H, W = pred_flows.shape

    original = images[0]
    target_flow = visualize_flow(target_flow_tensor, verbose=False)
    pred_flow = visualize_flow(pred_flows, verbose=False)

    # (1) Ground-truth dewarp
    with torch.no_grad():
        target_dewarped = apply_bm_doc3d(image_tensor, target_flow_tensor)
        pred_dewarped = apply_bm_doc3d(image_tensor, pred_flows)
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row titles
    axes[0, 0].set_ylabel('Target', fontsize=14, fontweight='bold', rotation=0, labelpad=40)
    axes[1, 0].set_ylabel('Predicted', fontsize=14, fontweight='bold', rotation=0, labelpad=40)
    
    # Column titles
    col_titles = ['Original', 'Dewarped', 'Flow']
    for i, title in enumerate(col_titles):
        axes[0, i].set_title(title, fontsize=12, fontweight='bold')
    
    # Plot target row
    axes[0, 0].imshow(prepare_tensor(original))
    axes[0, 1].imshow(prepare_tensor(target_dewarped))
    axes[0, 2].imshow(prepare_tensor(target_flow))
    
    # Plot predicted row
    axes[1, 0].imshow(prepare_tensor(original))
    axes[1, 1].imshow(prepare_tensor(pred_dewarped))
    axes[1, 2].imshow(prepare_tensor(pred_flow))
    
    # Remove ticks from all axes
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    plt.savefig(f"epoch_{epoch}_warps", dpi=300, bbox_inches='tight')
    
    # Save figure to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    # Convert to PIL Image
    pil_image = Image.open(buf)
    
    # Log to wandb
    wandb.log({
        f"epoch_{epoch}_warps": wandb.Image(pil_image, caption="Input | Warp | Flow")
    })