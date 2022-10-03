The following instruction is for reproducing our results of the S3DIS data. 

First download the S3DIS from the official [page](http://buildingparser.stanford.edu/dataset.html). 
Our preprocessed normals and oversegmentations for the S3DIS scenes can be downloaded here: [oversegmentation](https://datasets.d2.mpi-inf.mpg.de/box2mask/segment_labels.tar.gz) and [normals](https://datasets.d2.mpi-inf.mpg.de/box2mask/normals.tar.gz). 
Unzip the S3DIS data and the `normals` to `box2mask/data/Stanford3dDataset_v1.2_Aligned_Version/`. The structure of the unzipped data is as follows:

```
box2mask/data/Stanford3dDataset_v1.2_Aligned_Version/
└──  Area_1/                                    # Containing point cloud, segmentation information, normals, colors informations
    ├── hallway_1/ 
        ├── Annotations/ # Contains instances information 
            ├── door_2.txt  
            ├── floor_1.txt  
            ├── wall_2.txt
            ...
        ├── hallway_1.txt # Contains positions and colors of scene points
    ├── office_11/
        ...          
    ├── office_12/
        ...
    ...     
└──  Area_2/        
    ...   
└──  Area_3/
    ...
└──  Area_4/
    ...
└──  Area_5/
    ...
└──  Area_6/
    ...
└──  normals/
    ├── Area_4.office_7.npy
    ├── Area_5.office_36.npy
    ├── Area_1.office_25.npy
    ...
...
```

Run the following script to prepare the S3DIS dataset

```bash
mkdir -p ./data/s3dis/
python dataprocessing/prepare_s3dis.py --data_dir ./data/Stanford3dDataset_v1.2_Aligned_Version/
```

Uncompress the `segment_labels.tar.gz` file to `box2mask/data/s3dis/`

The preprocessed data and oversegmentation folders should be prepared as the following structure for our project:

```
box2mask/data/s3dis/
└──  Area_1/                                    # Containing point cloud, segmentation information, normals, colors informations
    ├── hallway_1.normals.instance.npy   
    ├── office_11.normals.instance.npy          
    ├── office_12.normals.instance.npy             
    ...   
└──  Area_2/
    ├── office_1.normals.instance.npy  
    ├── office_2.normals.instance.npy  
    ├── office_3.normals.instance.npy
    ...
└──  Area_6/
    ├── conferenceRoom_1.normals.instance.npy
    ├── copyRoom_1.normals.instance.npy
    ├── office_3.normals.instance.npy
    ...
└── segment_labels/                             # Containing the segmentation files of all scenes
    ├──learned_superpoin_graph_segmentations/
        ├── Area_4.office_7.npy
        ├── Area_5.office_36.npy
        ├── Area_1.office_25.npy
        ...
```

Here each `.normals.instance.npy` contains the point cloud, segmentations, colors and normals information. The information can be each extracted using the following script (note: instance labels is only used to get axis aligned bounding box information of each instance):

```python 
data = np.load ('box2mask/data/s3dis/Area_1/hallway_1.normals.instance.npy')
    
positions = data [:,:3].astype (np.float32)         # XYZ positions (N x 3)
colors = data [:,3:6].astype (np.float) / 255       # Point colors (N x 3)
normals = data [:,6:9].astype (np.float)            # Surface normals (N x 3)
semantics = data [:, -2].astype (np.int32)          # Semantic labels of points (N x 1)
instances = data [:, -1].astype (np.int32)          # Instance labels of points (N x 1)
```

You can train the model using `training.py` from the root folder. Each config file is of format s3dis_fold\[area_number\] which area_number indicate the area to be used as validation set and other areas to be used as training set. For example, to have area 5 as the validation set and other areas for training:

```bash
python models/training.py --config configs/s3dis_fold5.txt
```

To evaluate with the validation, run the following commands (producing the validation score as in Table 1 with Area 5):

```bash
python models/evaluation.py --config configs/s3dis_fold5.txt
```

You can also produce visualization by adding option `--produce_visualizations`. To choose a specific scene to process, provide a scene name with the option `--predict_specific_scene`. Each scene has the name in the following format `Area_[area_number].[room_name]` where `[area_number]` is a number from 1 to 6 and `[room_name]` the name of the room in the area. Producing result for a specific scene can be achived via `model/evaluation.py` with `--predict_specific_scene` option, see the example below:

```bash
python models/evaluation.py --config configs/s3dis_fold5.txt --predict_specific_scene Area_5.office_7 --produce_visualizations
```

Running the command above will produce the visualization of segmentation result in `experiments/s3dis_fold5/results/[checkpoint]/viz/Area_5.office_7` where `checkpoint` is the loaded checkpoint when running the script.