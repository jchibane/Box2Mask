The following instruction is for reproducing the experiments in Table. 7 in our paper. 

Follow the original ARKitScenes [instruction](https://github.com/apple/ARKitScenes/blob/main/DATA.md) to download the data (3dod dataset). 
The oversegmentation for ARKitScenes can be download here: [train](https://datasets.d2.mpi-inf.mpg.de/box2mask/segmented_train_clean.tar.gz) and [valid](https://datasets.d2.mpi-inf.mpg.de/box2mask/segmented_val_clean.tar.gz).
After you download the data and our prepared oversegmentations. The `Training` and `Validation` and oversegmentation folders should be prepared as the following structure for our project:

```
box2mask/data/ARKitScenes/3dod/
└── Training
    ├── 44358604                            # scene name
        ├── 44358604_3dod_annotation.json   # segmentation label of the scene
        ├── 44358604_3dod_mesh.ply          # mesh file 
        ├── 44358604_frames/                # Containing RGBD camera sequences 
    ├── 45662912
        ├── 45662912_3dod_annotation.json  
        ├── 45662912_3dod_mesh.ply  
        ├── 45662912_frames/
    ...
└── Validation/
    ├── 41069021
        ├── 41069021_3dod_annotation.json  
        ├── 41069021_3dod_mesh.ply  
        ├── 41069021_frames/
    ├──
    ...   
└── segmented_train_clean/
    ├── 47331587_3dod_mesh.0.010000.segs.json
    ├── 44358604_3dod_mesh.0.010000.segs.json
    ...
└── segmented_val_clean/
    ├── 41069021_3dod_mesh.0.010000.segs.json
    ...
```

Similar to the main experiment, you can train the model using `training.py` from the root folder::

```python
python models/training.py --config configs/arkitscenes.txt
```

To evaluate with the validation set (producing results like Table 2):

```python
python models/evaluation.py --config configs/arkitscenes.txt
```

You can also produce visualization by adding option `--produce_visualizations`. Producing result for a specific scene can be achived via `model/evaluation.py` with `--predict_specific_scene` option, see the example below:

```python
python models/evaluation.py --config configs/arkitscenes.txt --predict_specific_scene 42445429 --produce_visualizations
```

Running the command above will produce the visualization of segmentation result in `experiments/deb/results/[checkpoint]/viz/42445429` where `checkpoint` is the loaded checkpoint when running the script.