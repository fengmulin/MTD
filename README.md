# MTD
# RDB

Pull Pole Points to Text Contour by Magnetism: A Real-Time Scene Text Detector

## Environment
The environment, datasets, and usage are based on: [DBNet](https://github.com/MhLiao/DB)
```bash
conda create -n RDB python==3.9
conda activate RDB

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

git clone https://github.com/fengmulin/RDB.git
cd RDB/

echo $CUDA_HOME
cd assets/ops/dcn/
python setup.py build_ext --inplace

```

## Evaluate the performance
```
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/rdb/mpsc/res18.yaml --box_thresh 0.5 --resume workspace/mpsc/mpsc_res18
CUDA_VISIBLE_DEVICES=1 python eval.py experiments/rdb/total/res50.yaml --box_thresh 0.65 --polygon --resume workspace/total/total_res50
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/rdb/td500/res50.yaml --box_thresh 0.45 --resume workspace/td500/td500_res50
CUDA_VISIBLE_DEVICES=2 python eval.py experiments/rdb/ic15/res50.yaml --box_thresh 0.55 --resume workspace/ic15/ic15_res50
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/rdb/ctw/res50.yaml --box_thresh 0.5 --thresh 0.15 --polygon --resume workspace/ctw/ctw_res50
```

## Evaluate the speed 

```CUDA_VISIBLE_DEVICES=0 python eval.py experiments/rdb/mpsc/res18.yaml --box_thresh 0.5 --resume workspace/mpsc/mpsc_res18 --speed```


## Training

```CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py path-to-yaml-file --num_gpus 4```

## Acknowledgement
Thanks to [DBNet](https://github.com/MhLiao/DB) and [TextBPN++](https://github.com/GXYM/TextBPN-Plus-Plus) for a standardized training and inference framework. 

## Models
MTD trained models [Google Drive](https://drive.google.com/file/d/13A-oL912cyD0tKbqu3SQiAf4WaTf4gfp/view?usp=drive_link).








    
