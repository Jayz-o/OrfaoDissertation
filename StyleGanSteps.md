Protocol 2:
gans trained on ipad pro, iphone 7 plus, asus, samsung

protocol 3:
gan trained on Print and replay

for baseline, train gan on subject 90 and test with subject 75

Prepare Dataset
# Scaled down 256x256 resolution.
python dataset_tool.py --source=/tmp/images1024x1024 --dest=~/datasets/ffhq-256x256.zip --resolution=256x256
    

Try with same gamma, then, try increasing the value by 2x and 4x, and also decreasing it by 2x and 4x. Pick the lowest FID.
Reasonable --kimg at 5000, 25000 is default

1 gpu:
--cfg=stylegan3-r --gpus=1 --batch=32 --gamma=2 --batch-gpu=8 --snap=10 --kimg=5000

8 gpu:
--cfg=stylegan3-r --gpus=8 --batch=32 --gamma=2 --kimg=5000

Train 1 gpu:

python train.py --outdir=~/training-runs --data=~/datasets/afhqv2-512x512.zip --mirror=1 --cfg=stylegan3-r --gpus=1 --batch=32 --gamma=32 --batch-gpu=8 --snap=10 --kimg=5000

Train 4:
python train.py --outdir=~/training-runs --data=~/datasets/afhqv2-512x512.zip --mirror=1 --cfg=stylegan3-r --gpus=4 --batch=32 --gamma=32 --batch-gpu=4 --snap=20--kimg=5000
--cfg=stylegan3-r --gpus=4 --batch=32 --gamma=32 --batch-gpu=4 --snap=20

Train 8 gpu:

python train.py --outdir=~/training-runs --data=~/datasets/afhqv2-512x512.zip --mirror=1 --cfg=stylegan3-r --gpus=8 --batch=32 --gamma=32 --kimg=5000

generate Images:
python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl