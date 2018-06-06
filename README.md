# TSN-Pytorch-gesture-recognition
This a gesture recognition experiment in vedio for a logitics project.We use TSN model to discern action from a video in trunk.We can get driver's driving video by camera in the trunk to judge whether the driver has illegal manner when driving.The origin paper can be found [here](https://arxiv.org/abs/1506.01497). For more detail about the paper and code, see this [blog][1]


***
# setup
- requirements: pytorch2.3, opencv-python, opencv-c++
we use opencv-c++ to extract optical flow
# demo
- put your images in data/demo, the results will be saved in data/results, and run demo in the root 
```shell
python ..//tools/demo.py
```
***
# training
## prepare data
- We use own private data to train tsn model, but sorry we couldn't open these data. You can use International public dataset,such as UFC101,hmdb51 e.g.
- Second, you need to download pretrain models adn put it in models. 
- We use three input modes to train our models,for RGB mode, optical mode, RGB-Diff mode.

for RGB mode:
```bash
python main.py ucf101 RGB <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ 
```

For flow models:

```bash
python main.py ucf101 Flow <ucf101_flow_train_list> <ucf101_flow_val_list> \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 128 -j 8 --dropout 0.7 \
   --snapshot_pref ucf101_bninception_ --flow_pref flow_  
```

For RGB-diff models:

```bash
python main.py ucf101 RGBDiff <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
   --arch BNInception --num_segments 7 \
   --gd 40 --lr 0.001 --lr_steps 80 160 --epochs 180 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ 
```

## Testing

After training, there will checkpoints saved by pytorch, for example `ucf101_bninception_rgb_checkpoint.pth`.

Use the following command to test its performance in the standard TSN testing protocol:

```bash
python test_models.py ucf101 RGB <ucf101_rgb_val_list> ucf101_bninception_rgb_checkpoint.pth \
   --arch BNInception --save_scores <score_file_name>

```

Or for flow models:
 
```bash
python test_models.py ucf101 Flow <ucf101_rgb_val_list> ucf101_bninception_flow_checkpoint.pth \
   --arch BNInception --save_scores <score_file_name> --flow_pref flow_

```
