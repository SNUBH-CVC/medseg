## MedSeg

## Installation
Install [SegMamba](https://github.com/ge-xing/SegMamba).
```bash
# casual-conv1d
cd 3rdparty/casual-conv1d
python setup.py install --user

# mamba
cd 3rdparty/mamba
python setup.py install --user
```

Install other requirements.
```
pip install -r requirements.txt
```

## Dataset
- ImageCAS

## Benchmarks
### Architecture
- UNet
- SegMamba
- UNETR
- SwinUNETR

### Feature
- skeletonize layer

### Loss
- clDice 
- dice 
- topological 

## Train
```
python tools/train.py [cfg_path]
```

## Test
```
python tools/test.py [run_id] --mlflow_tracking_uri [mlflow_tracking_uri]
```
