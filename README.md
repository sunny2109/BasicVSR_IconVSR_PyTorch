##  Dependencies and Installation
- The code is based on the old version of [BasicSR](https://github.com/xinntao/BasicSR/tree/update_niqe), Please install the BasicSR framework first.
- Pytorch=1.51

##  Training
```bash
cd ./code

CUDA_VISIBLE_DEVICES=0,1 python basicsr/train.py -opt options/train/BasicVSR/train_BasicVSR.yml

CUDA_VISIBLE_DEVICES=0,1 python basicsr/train.py -opt options/train/BasicVSR/train_IconVSR.yml
```
## Testing
```bash
cd ./code

CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/BasicVSR/test_BasicVSR_REDS.yml

CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/BasicVSR/test_BasicVSR_Vid4.yml
```

## PSNR/SSIM Results
It takes about 5 days to train the BasicVSR/IconVSR model with the [REDS](https://seungjunnah.github.io/Datasets/reds) dataset on 2 V100 GPUs.

| Dataset(BI) | BasicVSR (paper) | Ours |IconVSR_w/o Refill (paper) | Ours |
| :----- | :-----: | :-----: | :-----: | :-----: | 
| REDS4 | 31.42/0.8909 | 31.409/0.8907| 31.60 | 31.6026 |
| Vid4 | 27.24/0.8251 |27.269/0.8311| - | - |

- Pretrained models and SR results can be downloaded [Here](https://drive.google.com/file/d/1zE_EJ9S67GSL_FcaaeOZTfZdHyofoZzY/view?usp=sharing).
