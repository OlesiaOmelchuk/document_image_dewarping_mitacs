import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import time
import wandb
from types import SimpleNamespace

from model import FlowGenerator
from dataset import Doc3DDataset
from loss import FlowLoss
from visualization import visualize_epoch_results


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


if __name__ == '__main__':
    # Set your API key and login to W&B
    WANDB_API_KEY = "YOUR_WANDB_API_KEY"
    wandb.login(key=WANDB_API_KEY)

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
    def collate_fn(batch):
        # remove failed samples
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None  # skip empty batch
        return torch.utils.data.dataloader.default_collate(batch)

    dataset = Doc3DDataset(root_dir=args.data_root, target_size=(448, 448))

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    print("Train size:", train_size)
    print("Validation size:", val_size)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

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
