## Reproduce Augmented BBs Experiments

The following instruction is for reproducing the experiments in Fig. 7 in our paper. We use seeds to generate the same set of augmented data in every runs during training. The config files of these experiments are in `box2mask/configs/`. Name of each config is either `scannet_dropout[percentage]` (`percentage` is the boxes of boxes that are missing) or `scannet_noisy[sigma]` (`sigma` is the variance of the noise applied to each dimension).


Similar to the main experiment, you can train the model using the augmented bounding boxes like the example bellow:

```
python models/training.py --config configs/scannet_noisy1.txt
```

To evaluate with the validation set:

```
python models/evaluation.py --config configs/scannet_noisy1.txt
```

## Augmented Data

We also store our augmented BBs as npy files. The following script will download and extract the data to `data/augmented_BBs/scannet_augmented_boxes_data/`
```
cd data/augmented_BBs/
wget https://datasets.d2.mpi-inf.mpg.de/box2mask/scannet_augmented_boxes_data.tar.gz
tar -xvf scannet_augmented_boxes_data.tar.gz
```

The files are organized as follow:

```shell
<data_name>
|-- <scanId>.npy
```

where `<data_name>` is `dropout[percentage]` (missing bounding box labels data, `percentage` can be 1, 2, 5 or 10) or `noisy[sigma]` (noisy label data, `sigma` can be 2, 4, 10 or 20).
Each .npy file contains list of min corners and max corners of the bounding boxes as well as the semantic ids.

We provide script to visualize the bounding box for a scene in the data. The command bellow will produce an interactive visualization server in `data/augmented_BBs/visualize/`. 

```
cd data/augmented_BBs/
python visualize_bbs_data.py --data noisy1 --scene_name scene0293_00 --data_path data/augmented_BBs/scannet_augmented_boxes_data/
```
Use the command bellow to start the visualization server:

```
cd data/augmented_BBs/visualize/
python -m http.server 6008
```
