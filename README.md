## MedSeg

## Installation
```bash
# casual-conv1d
cd 3rdparty/casual-conv1d
python setup.py install --user

# mamba
cd 3rdparty/mamba
python setup.py install --user

pip install -r requirements.txt
```

## Dataset
- ImageCAS

## Benchmarks
### Base architecture
- UNet3D
- SegMamba
- UNETR
- SwinUNETR

### Additional feature
- skeletonize layer

### Loss
- clDice 
- dice 
- topological 

## Train
`python tools/train.py`