# Code for Paper "Active Kinematic Modelling for Precise Manipulation of Unseen Articulated Objects"

## File Structure
```
/assets
/baselines
/cfgs
/embodied_analogy
/scripts
/third_party
setup.py            # For Installation of Our Package
```

## Installation
We first provide the conda environment setup to run our method, as shown in `/Embodied_Analogy/scripts/prepare/create_akm_env.sh`. Conda Environments of baseline methods `GeneralFlow` and `GAPartnet` can be installed on the basis of our `AKM` environment, as shown in `create_gflow_env.sh` and `create_gpnet_env.sh`.

## Preparation
In this section we show how to prepare the required dataset and pretrained model checkpoints from third-party methods.

The pretrained model checkpoints are stored in `/assets/ckpts` folder, as,
```
/assets/ckpts
├── anygrasp
│   └── checkpoint_detection.tar
├── fastSAM
│   ├── FastSAM-s.pt
│   ├── FastSAM-x.pt
│   └── FastSAM-X.pt
├── gapartnet
│   ├── all_best.ckpt
│   └── release.ckpt
├── generalFlow
│   └── kpst_hoi4d
│       ├── ScaleGFlow-B
│       ├── ScaleGFlow-B.tar.gz
│       ├── ScaleGFlow-L
│       ├── ScaleGFlow-L.tar.gz
│       ├── ScaleGFlow-S
│       └── ScaleGFlow-S.tar.gz
├── grounding_dino
│   └── groundingdino_swint_ogc.pth
├── gsnet
│   ├── minkuresunet_kinect.tar
│   └── minkuresunet_realsense.tar
└── sam2
    └── sam2.1_hiera_large.pt
```
the download link can be refered to the respective links.
[AnyGrasp](https://github.com/graspnet/anygrasp_sdk)
[gapartnet](https://github.com/PKU-EPIC/GAPartNet)
[generalFlow](https://github.com/michaelyuancb/general_flow)
[grounding_dino](https://github.com/IDEA-Research/GroundingDINO)
[gsnet](https://github.com/graspnet/graspness_unofficial)
[sam2](https://github.com/facebookresearch/sam2)
[fastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)

The dataset used in our simulation environment is from [RGBManip](https://github.com/hyperplane-lab/RGBManip) and can be downloaded through [this_link](https://drive.google.com/file/d/154g8SzGFWOcLLjes40aTXFoREX47-ZPk/view), unzip the downloaded dataset.zip in `/assets/dataset` folder, and structure as:
```
/assets/dataset
├── chair
├── mugs
├── one_door_cabinet
├── one_drawer_cabinet
└── pots
```

The [RAM](https://github.com/yxKryptonite/RAM_code) affordance memory used in our method can be download through [this_link](https://drive.google.com/file/d/16cEIj8JHZ8KkGiRRub_qxdZoKB45ZXsa/view) and unzip as:
```
/assets/RAM_memory
├── customize
├── droid
├── HOI4D
└── memory_data.zip
```

Then run the `pre_extract_dift_feat.py` scripts to pre-extract [DIFT](https://github.com/Tsingularity/dift) feature for the reference images in the RAM affordance memory to accelerate the exploration stage in our method:
```
cd /Embodied_Analogy/scripts/prepare
conda activate akm
python pre_extract_dift_feat.py
```
After running this script, there should be an additional `*_new_sd.pkl` file in each memory directory:
```
assets/RAM_memory/customize/open_the_dishwasher
├── open_the_dishwasher.json
├── open_the_dishwasher_new.pkl
├── open_the_dishwasher_new_sd.pkl  (pre-extracted sd feature)
└── vis
```

## Test our method in Simulation
### run test script
### Evaluate result
### Aggregate Analysis from multiple run
## Test our method in Real-World 
