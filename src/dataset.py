
import os
import torch
from torch.utils.data import Dataset
import hdf5storage as h5
import cv2
import numpy as np
from PIL import Image

    
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