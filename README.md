# Deep Learning-Based Image Compression and Transmission for Improved Quality and Reduced Bandwidth Requirements

Authors: Suho Yang & Naufal Shidqi

## About
Pytorch implementation of KAIST Spring 2023 CS546 - Wireless Mobile Internet and Security final project.
This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [STF](https://github.com/Googolxx/STF). 
We modify the evaluation scripts for transmitting through Socket API.
Our proposed models (CSTF) are provided in `compressai/models`.


## Installation

Install [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the packages required for development.
Make sure you have gcc with c++17 installed.
```bash
# Install libraries
conda create -n compress python=3.7
conda activate compress
pip install compressai
pip install pybind11
pip install torchinfo
pip install tensorboard

# Clone repository and install CompressAI environment packages
git clone https://github.com/nshidqi/CSTF.git cstf
cd cstf
pip install -e .
pip install -e '.[dev]'
```

## Dataset
The script for downloading [OpenImages](https://github.com/openimages) is provided in `downloader_openimages.py`. Please install [fiftyone](https://github.com/voxel51/fiftyone) first. For Kodak image dataset, can be downloaded [here](http://r0k.us/graphics/kodak/).

## Training
We use `train.py` for training the models.
```bash
usage: train.py [-h] [-log LOG_PATH] [-exp EXP_NAME]
                [-m {stf,cstf_simple,cstf_general,cstf_general_321,cstf_general_321_embed_742,cstf_general_embed_742,cstf_simple_RPE,cstf_general_window_2_2_2_2,cstf_general_window_4_4_4_4,cnn}]
                -d DATASET [-e EPOCHS] [-lr LEARNING_RATE] [-n NUM_WORKERS]
                [--loss LOSS] [--lambda LMBDA] [--batch-size BATCH_SIZE]
                [--test-batch-size TEST_BATCH_SIZE]
                [--aux-learning-rate AUX_LEARNING_RATE]
                [--patch-size PATCH_SIZE PATCH_SIZE] [--cuda] [--save]
                [--save_path SAVE_PATH] [--seed SEED]
                [--clip_max_norm CLIP_MAX_NORM] [--checkpoint CHECKPOINT]

Example training script.

optional arguments:
  -h, --help            show this help message and exit
  -log LOG_PATH, --log_path LOG_PATH
                        log path
  -exp EXP_NAME, --exp_name EXP_NAME
  -m {stf,cstf_simple,cstf_general,cstf_general_321,cstf_general_321_embed_742,cstf_general_embed_742,cstf_simple_RPE,cstf_general_window_2_2_2_2,cstf_general_window_4_4_4_4,cnn}, --model {stf,cstf_simple,cstf_general,cstf_general_321,cstf_general_321_embed_742,cstf_general_embed_742,cstf_simple_RPE,cstf_general_window_2_2_2_2,cstf_general_window_4_4_4_4,cnn}
                        Model architecture (default: stf)
  -d DATASET, --dataset DATASET
                        Training dataset
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 100)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate (default: 0.0001)
  -n NUM_WORKERS, --num-workers NUM_WORKERS
                        Dataloaders threads (default: 30)
  --loss LOSS           Criterion Loss Function (default: mse)
  --lambda LMBDA        Bit-rate distortion parameter (default: 0.01)
  --batch-size BATCH_SIZE
                        Batch size (default: 16)
  --test-batch-size TEST_BATCH_SIZE
                        Test batch size (default: 64)
  --aux-learning-rate AUX_LEARNING_RATE
                        Auxiliary loss learning rate (default: 0.001)
  --patch-size PATCH_SIZE PATCH_SIZE
                        Size of the patches to be cropped (default: (256,
                        256))
  --cuda                Use cuda
  --save                Save model to disk
  --save_path SAVE_PATH
                        Where to Save model
  --seed SEED           Set random seed for reproducibility
  --clip_max_norm CLIP_MAX_NORM
                        gradient clipping max norm (default: 1.0
  --checkpoint CHECKPOINT
                        Path to a checkpoint
```

### Example of training baseline model:
```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py -d datasets/openimages/ -e 25 --batch-size 16 --save --save_path checkpoints/demo_baseline.pth.tar -m stf --cuda --lambda 4.58 --loss ms-ssim --seed 2023
```
### Example of training our model:
```bash
CUDA_VISIBLE_DEVICES=3,4 python train.py -d datasets/openimages/ -e 25 --batch-size 16 --save --save_path ./checkpoints/demo_ours.pth.tar -m cstf_general_321 --cuda --lambda 4.58 --loss ms-ssim --seed 2023
```

## Evaluation

We use `compressai.utils.eval_model` for training the models.

```bash
usage: __main__.py [-h] [--socket_role SOCKET_ROLE] [-d DATASET]
                   [-r RECON_PATH] -a
                   {stf,cstf_simple,cstf_general,cstf_general_321,cstf_general_321_embed_742,cstf_general_embed_742,cstf_simple_RPE,cstf_general_window_2_2_2_2,cstf_general_window_4_4_4_4,cnn}
                   [-c {ans}] [--cuda] [--half] [--entropy-estimation] [-v] -p
                   [PATHS [PATHS ...]] [--crop]

optional arguments:
  -h, --help            show this help message and exit
  --socket_role SOCKET_ROLE
                        socket role {server, client, none}
  -d DATASET, --dataset DATASET
                        dataset path
  -r RECON_PATH, --recon_path RECON_PATH
                        where to save recon img
  -a {stf,cstf_simple,cstf_general,cstf_general_321,cstf_general_321_embed_742,cstf_general_embed_742,cstf_simple_RPE,cstf_general_window_2_2_2_2,cstf_general_window_4_4_4_4,cnn}, --architecture {stf,cstf_simple,cstf_general,cstf_general_321,cstf_general_321_embed_742,cstf_general_embed_742,cstf_simple_RPE,cstf_general_window_2_2_2_2,cstf_general_window_4_4_4_4,cnn}
                        model architecture
  -c {ans}, --entropy-coder {ans}
                        entropy coder (default: ans)
  --cuda                enable CUDA
  --half                convert model to half floating point (fp16)
  --entropy-estimation  use evaluated entropy estimation (no entropy coding)
  -v, --verbose         verbose mode
  -p [PATHS [PATHS ...]], --path [PATHS [PATHS ...]]
                        checkpoint path
  --crop                center crop 256 256 to test
```
### Example of evaluating model without socket (offline compression-decompression)
```bash
CUDA_VISIBLE_DEVICES=3 python -m compressai.utils.eval_model -d datasets/kodak_1/ -r reconstruction/output_image_baseline/ -a stf -p checkpoints/demo_baseline.pth.tar --cuda --socket_role none
```

### Example of evaluating model with socket API (online compression-decompression)
First, run this command in a device as server (receive and decompress):
```bash
CUDA_VISIBLE_DEVICES=3 python -m compressai.utils.eval_model -d datasets/kodak_1/ -r reconstruction/output_image_baseline/ -a stf -p checkpoints/demo_baseline.pth.tar --cuda --socket_role server
```
Second, run this command in a device as client (compress and transmit):
```bash
CUDA_VISIBLE_DEVICES=3 python -m compressai.utils.eval_model -d datasets/kodak_1/ -r reconstruction/output_image_baseline/ -a stf -p checkpoints/demo_baseline.pth.tar --cuda --socket_role server
```



## Related links
 * CompressAI: https://github.com/InterDigitalInc/CompressAI
 * Swin-Transformer: https://github.com/microsoft/Swin-Transformer
 * Symmetrical Transformer: https://github.com/Googolxx/STF
 * CSwin-Transformer: https://github.com/microsoft/CSWin-Transformer
 * Range Asymmetric Numeral System code from Fabian 'ryg' Giesen: https://github.com/rygorous/ryg_rans
 * Kodak Images Dataset: http://r0k.us/graphics/kodak/
 * Open Images Dataset: https://github.com/openimages
 * fiftyone: https://github.com/voxel51/fiftyone
