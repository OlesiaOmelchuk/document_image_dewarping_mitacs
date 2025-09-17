# Transformer-UNet-based Architecture for Document Dewarping: Training and Downstream Evaluation on Layout Analysis Task
Document image dewarping remains challenging due to complex distortions and reliance on the synthetic datasets for training. We propose a Transformer-based Encoder with a UNet-style Decoder architecture that predicts flow maps for direct document dewarping. To reduce reliance on paired supervision, we explore an unsupervised cycle-consistency paradigm and analyze its limitations. Beyond standard OCR-based and perceptual similarity metrics, we introduce a novel evaluation angle: assessing how dewarping quality influences the downstream document layout analysis task. Experiments on the Doc3D training set and evaluations on the DIR300 and DocUNet benchmarks demonstrate that the proposed architecture achieves competitive performance in the supervised setting, while unsupervised training remains a challenging open problem. Importantly, our results reveal the strong dependency of layout analysis accuracy on image 'flatness', emphasizing the need for a robust dewarping solution in document understanding. 

## Installation
```bash
pip install -r requirements.txt
pip install opencv-python
```

## Project tree
```
├── notebooks                                       # Kaggle notebooks
│   ├── docunet-benchmark.ipynb                     # DocUNet benchmark code
│   ├── end-to-end-inference.ipynb                  # Dewarping, OCR, layout inference
│   ├── kaggle-supervised-dewarping-training.ipynb  # Kaggle model training code
│   ├── model-inference.ipynb                       # Dewarping model inference
│   ├── tesseract-ocr.ipynb                         # OCR code
│   ├── visualizations.ipynb                        # Visualizations for paper
│   └── yolov10-layout-analysis.ipynb               # Layout analysis code 
├── README.md
├── requirements.txt
├── run_supervised_training.sh                      # Compute Canada job script
└── src                                             # Source code
    ├── dataset.py                                  # Doc3D dataset class
    ├── loss.py                                     # Training losses
    ├── model.py                                    # Dewarping model architecture
    ├── train_supervised_dewarper_v2.py             # Compute Canada training script
    ├── train.py                                    # Refactored training script
    └── visualization.py                            # Debug training visualizations                               
```
## Model Training
```train_supervised_dewarper_v2.py``` is the main script used for model training. It contains all the necessary implementation code in one file and was used directly for training setup on Compute Canada (see the bash script ```run_supervised_training.sh```). 

```train.py``` is just a refactored version of the ```train_supervised_dewarper_v2.py ``` with all the implementation code being splitted into separate files and stored in the [src](./src/) directory.

Before running the script, make sure to insert you wandb API key and modify training configurations:
```python
if __name__ == '__main__':
    # Set your API key and login to W&B
    WANDB_API_KEY = "YOUR_WANDB_API_KEY"
    wandb.login(key=WANDB_API_KEY)

    # ---------------------------
    # Configuration
    # ---------------------------
    config = {
        'data_root': os.path.join("/", "home", "olesiao", "scratch", "olesiao", "doc3d"),
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
...
```

## Evaluation
For evaluation we selected DIR300 and DocUNet benchmarks. Implementation of the evaluation code was based on the [DocGeoNet](https://github.com/fh2019ustc/DocGeoNet) repository. For CER and ED metrics, python code was used; for MS-SSIM and LD - MATLAB.

## References
```
@inproceedings{feng2022geometric,
  title={Geometric representation learning for document image rectification},
  author={Feng, Hao and Zhou, Wengang and Deng, Jiajun and Wang, Yuechen and Li, Houqiang},
  booktitle={European Conference on Computer Vision},
  pages={475--492},
  year={2022}
}
```