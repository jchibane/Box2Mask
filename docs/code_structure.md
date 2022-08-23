# Code structure
**configs/**, Includes the config files to run models

**data/**, Storing datasets (eg. data/scannet/ or data/ARKitScenes)

**config_loader.py**, Defines all hyper-parameters of the model


**dataprocessing**
  - **dataprocessing/augmentation.py**, Defines augmentation code
  - **dataprocessing/scannet.py**, Reads on train/test/val scenes of scannet
  - **dataprocessing/arkitscenes.py**, Reads on train/test/val scenes of Arkitscenes


**models**
  - **models/dataloader.py**, Reads and preprocesses data and prepare tensor batches
    - **class ScanNet**, Reads and preprocesses Scannet scenes
        - **approx_association()**, Finds the associations of points using GT bounding boxes 
        - **__getitem__()**, Preprocesses the scenes, returns model inputs and labels
    - **class ARKitScenes**, Reads and preprocess ArkitScenes scenes
        - **approx_association()**, Finds the associations of points using GT bounding boxes 
        - **__getitem__()**, Preprocesses the scenes, returns model inputs and labels
    - **collate_fn**, Collates preprocessed scenes into tensor batches

  - **models/detection_net.py**, Defines the network
    - **class SelectionNet**, Define the main network and network heads
        - **detection2mask()**, Converts box proposals into final instance mask 
        - **get_prediction()**, Gets prediction from the network heads
  - **models/evaluation.py**, Evaluates Scannet and ArkitScenes predictions. Can be run with:  `python models/evaluation.py  --config configs/[config_name].txt`
    - **arkitscenes_eval()**, Approximates oriented bounding boxes from instance predictions and computes  detection quality using the AP score
    - **scannet_eval()**, Computes Scannet prediction scores in terms of AP, AP50 and AP25
  - **models/iou_nms.py**, Defines the Non-Maximum Clustering clustering
    - **NMS_clustering()**, Non-Maximum Clustering algorithm (as in Sec.3 and Sec. 4 in the main paper)
  - **models/resnet.py**, Some utilities for making the U-Net model
  - **models/training.py**, Defines the training code, can be run witch: `python models/training.py  --config configs/[config_name].txt`
  - **models/model.py**, Defines and computes the losses for each epoch
    - **compute_loss_detection()**, Compute each loss and the weighted joint losses for the network optimization

**utils/**, Contains some low-level utilities