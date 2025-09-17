import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import hdf5storage as h5
import cv2
import numpy as np
from einops import rearrange
import time
import argparse
from torchvision import transforms
import torchvision.utils as vutils
import wandb  # Optional for logging
import matplotlib.pyplot as plt


# ---------------------------
# Model Architecture (Your Transformer+U-Net)
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, p=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=False)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Linear(int(dim*mlp_ratio), dim)
        )
    def forward(self, x):  # x: [HW,B,D]
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x

class MultiStageTransformerEncoder(nn.Module):
    def __init__(self, img_channels=3, embed_dims=[64,128,256], patch_sizes=[8,16,2], depths=[2,2,2], heads=[2,4,8]):
        super().__init__()
        self.stages = nn.ModuleList()
        self.embed_dims = embed_dims
        for i, d in enumerate(embed_dims):
            in_ch = img_channels if i == 0 else embed_dims[i-1]
            self.stages.append(nn.ModuleDict({
                "proj": nn.Conv2d(in_ch, d, kernel_size=patch_sizes[i], stride=patch_sizes[i]),
                "blocks": nn.ModuleList([TransformerBlock(d, heads[i]) for _ in range(depths[i])])
            }))
    def forward(self, x):
        skips = []
        for s in self.stages:
            x = s["proj"](x)             # [B,D,h,w]
            B, D, h, w = x.shape
            x_seq = rearrange(x, "b d h w -> (h w) b d")
            for blk in s["blocks"]:
                x_seq = blk(x_seq)
            x = rearrange(x_seq, "(h w) b d -> b d h w", h=h, w=w)
            skips.append(x)
        return skips  # [low-res ... high-res]

class UNetDecoder(nn.Module):
    def __init__(self, embed_dims=[64,128,256], out_ch=2):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dims[2], embed_dims[1], 2, 2)
        self.c1  = nn.Sequential(nn.Conv2d(embed_dims[1]*2, embed_dims[1], 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(embed_dims[1], embed_dims[1], 3, padding=1), nn.ReLU(True))
        self.up2 = nn.ConvTranspose2d(embed_dims[1], embed_dims[0], 2, 2)
        self.c2  = nn.Sequential(nn.Conv2d(embed_dims[0]*2, embed_dims[0], 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(embed_dims[0], embed_dims[0], 3, padding=1), nn.ReLU(True))
        self.up3 = nn.ConvTranspose2d(embed_dims[0], embed_dims[0]//2, 2, 2)
        self.c3  = nn.Sequential(nn.Conv2d(embed_dims[0]//2, embed_dims[0]//2, 3, padding=1), nn.ReLU(True))
        self.out = nn.Conv2d(embed_dims[0]//2, out_ch, 1)
    def forward(self, skips):
        x = skips[-1]
        x = self.up1(x)
        s1 = F.interpolate(skips[1], size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s1], dim=1); x = self.c1(x)

        x = self.up2(x)
        s0 = F.interpolate(skips[0], size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s0], dim=1); x = self.c2(x)

        x = self.up3(x); x = self.c3(x)
        return self.out(x)

class FlowGenerator(nn.Module):
    """Predicts flow to transform input between domains."""
    def __init__(self, img_channels=3, max_disp=48.0):
        super().__init__()
        self.enc = MultiStageTransformerEncoder(img_channels=img_channels)
        self.dec = UNetDecoder()
        self.max_disp = max_disp
    def forward(self, x):
        B, C, H, W = x.shape
        skips = self.enc(x)
        flow = self.dec(skips)
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        # constrain displacement magnitude for stability
        # flow = torch.tanh(flow) * self.max_disp
        flow = flow * 10.0  # EXPERIMENT: 10x amplification for visibility
        return flow
        

from PIL import Image

def safe_imread(img_path):
    image = cv2.imread(img_path)
    if image is None:
        try:
            # fallback to PIL
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                image = np.array(im)[:, :, ::-1]  # convert RGB→BGR so cv2 is consistent
        except Exception as e:
            raise RuntimeError(f"Could not read image {img_path}: {e}")
    return image

    
class Doc3DDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(448, 448), align_corners=True):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.align_corners = align_corners
        
        # Collect all image paths
        self.image_paths = []
        img_dir = os.path.join(root_dir, 'img')
        print("ROOT DIR EXISTS:", os.path.exists(root_dir))
        print("IMG DIR EXISTS:", os.path.exists(img_dir))

        for folder in os.listdir(img_dir):
            folder_path = os.path.join(img_dir, folder)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.endswith('.png'):
                        self.image_paths.append((folder, fname))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        folder, fname = self.image_paths[idx]
        base_name = fname[:-4]  # remove .png extension

        img_path = os.path.join(self.root_dir, 'img', folder, fname)
        bm_path  = os.path.join(self.root_dir, 'bm', folder, base_name + '.mat')

        try:
            # --- Load image ---
            image = safe_imread(img_path)
            if image is None:
                raise RuntimeError(f"Image is None: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.target_size)
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

            # --- Load backward map ---
            bm_data = h5.loadmat(bm_path)
            backward_map = bm_data['bm'].astype(np.float32)  # (H,W,2)

            H_orig, W_orig = backward_map.shape[:2]
            H_tgt, W_tgt = self.target_size

            if (H_orig, W_orig) != (H_tgt, W_tgt):
                # Resize & scale BM properly
                scale_x = W_tgt / W_orig
                scale_y = H_tgt / H_orig
                bm_resized = np.zeros((H_tgt, W_tgt, 2), dtype=np.float32)
                bm_resized[..., 0] = cv2.resize(backward_map[..., 0], (W_tgt, H_tgt)) * scale_x
                bm_resized[..., 1] = cv2.resize(backward_map[..., 1], (W_tgt, H_tgt)) * scale_y
                backward_map = bm_resized

            bm_pix = torch.from_numpy(backward_map).permute(2, 0, 1)  # (2,H,W)

            if self.align_corners:
                norm_x = (bm_pix[0] / (W_tgt - 1)) * 2 - 1
                norm_y = (bm_pix[1] / (H_tgt - 1)) * 2 - 1
            else:
                norm_x = (2 * bm_pix[0] + 1) / W_tgt - 1
                norm_y = (2 * bm_pix[1] + 1) / H_tgt - 1
            bm_norm = torch.stack([norm_x, norm_y], dim=0)

            return image_tensor, bm_pix.float(), bm_norm.float()

        except Exception as e:
            print(f"[WARNING] Skipping sample {img_path} (error: {e})")
            return None



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

from io import BytesIO
from PIL import Image

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

    # ---------------------------
# Training Function
# ---------------------------
def train_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch, log_wandb):
    model.train()
    total_loss, total_l1, total_tv, total_jac = 0, 0, 0, 0
    start_time = time.time()
    
    for batch_idx, (images, target_flows, target_flows_norm) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        target_flows = target_flows.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            pred_flows = model(images)   # (B,2,H,W), normalized
            loss, l1_loss, tv_loss_val, jac_loss_val = criterion(pred_flows, target_flows)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate stats
        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_tv += tv_loss_val.item()
        total_jac += jac_loss_val.item()
        
        # Progress logging
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch} | Batch {batch_idx}/{len(dataloader)}] "
                  f"Loss={loss.item():.4f} | L1={l1_loss.item():.4f} | "
                  f"TV={tv_loss_val.item():.4f} | Jac={jac_loss_val.item():.4f} "
                  f"| Time={elapsed:.1f}s")
            
            if log_wandb and wandb.run is not None:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_l1_loss": l1_loss.item(),
                    "batch_tv_loss": tv_loss_val.item(),
                    "batch_jac_loss": jac_loss_val.item(),
                    "batch_idx": batch_idx + epoch * len(dataloader)
                })
    
    n = len(dataloader)
    return total_loss/n, total_l1/n, total_tv/n, total_jac/n

# ---------------------------
# Validation Function
# ---------------------------
def validate(model, dataloader, criterion, device, epoch, log_wandb):
    model.eval()
    total_loss, total_l1, total_tv, total_jac = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, target_flows, target_flows_norm in dataloader:
            images = images.to(device, non_blocking=True)
            target_flows = target_flows.to(device, non_blocking=True)
            
            pred_flows = model(images)
            loss, l1_loss, tv_loss_val, jac_loss_val = criterion(pred_flows, target_flows)
            
            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_tv += tv_loss_val.item()
            total_jac += jac_loss_val.item()
    
    n = len(dataloader)

    if log_wandb and wandb.run is not None:
        visualize_epoch_results(model, dataloader, device, epoch+1)
    
    return total_loss/n, total_l1/n, total_tv/n, total_jac/n


import wandb

# Set your API key and login to W&B
WANDB_API_KEY = "7e96fbfc04f6dc1cb4dc7d4aca5259056805f90c"
wandb.login(key=WANDB_API_KEY)

from types import SimpleNamespace

# ---------------------------
# Configuration
# ---------------------------
config = {
    'data_root': os.path.join("/", "home", "olesiao", "scratch", "olesiao", "doc3d"),  # required
    'batch_size': 32,
    'epochs': 100,
    'lr': 1e-4,
    'save_dir': 'checkpoints',
    'resume': os.path.join("/", "home", "olesiao", "projects", "def-saadi", "olesiao", "supervised_runs", "checkpoints", "checkpoint_epoch_24.pth"),
    'max_disp': 48.0,
    'tv_weight': 0.001,
    'jac_weight': 0.0002,
    'use_wandb': True
}

# Convert to namespace for dot notation access
args = SimpleNamespace(**config)

# -----------------------------
# Setup
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize wandb
if args.use_wandb:
    wandb.init(
        project='supervised-dewarping-project', 
        config=args,
        name='doc3d-100k-compute_canada',
        resume=True,
        id='1fq8gatq'
    )

# Model
model = FlowGenerator(max_disp=args.max_disp).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Resume checkpoint
start_epoch = 0
if args.resume:
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'Resumed from epoch {start_epoch}')

# -----------------------------
# Dataset / Dataloader
# -----------------------------
dataset = Doc3DDataset(root_dir=args.data_root, target_size=(448, 448))

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
print("Train size:", train_size)
print("Validation size:", val_size)

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

def collate_fn(batch):
    # remove failed samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # skip empty batch
    return torch.utils.data.dataloader.default_collate(batch)

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=4, pin_memory=True, collate_fn=collate_fn
)

# -----------------------------
# Optimizer / Loss
# -----------------------------
criterion = FlowLoss(tv_weight=args.tv_weight, jac_weight=args.jac_weight)
scaler = GradScaler()

os.makedirs(args.save_dir, exist_ok=True)

# -----------------------------
# Training Loop
# -----------------------------
best_val_loss = float('inf')

for epoch in range(start_epoch, args.epochs):
    print(f'\nEpoch {epoch+1}/{args.epochs}')

    # Train
    train_loss, train_l1, train_tv, train_jac = train_epoch(
        model, train_loader, optimizer, criterion, device, scaler, epoch, args.use_wandb
    )

    # Validate
    val_loss, val_l1, val_tv, val_jac = validate(model, val_loader, criterion, device, epoch, args.use_wandb)

    print(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    print(f'  Train L1: {train_l1:.4f} | TV: {train_tv:.4f} | Jac: {train_jac:.4f}')
    print(f'  Val L1: {val_l1:.4f}')

    # wandb logging
    if args.use_wandb:
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_l1_loss': train_l1,
            'train_tv_loss': train_tv,
            'train_jac_loss': train_jac,
            'val_loss': val_loss,
            'val_l1_loss': val_l1,
            'val_tv_loss': train_tv,
            'val_jac_loss': train_jac,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
        print(f'New best model saved with val loss: {val_loss:.4f}')

print('Training completed!')
if args.use_wandb:
    wandb.finish()
