# Official PyTorch implementation for OSDA-ST

## How to run

### Setup Enviorment

We used python 3.8.5.

```shell
python -m venv ~/venv/bus
source ~/venv/bus/bin/activate
```

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.

### Setup Datasets
Download the potsdam and vaihingen datasets from ISPRS:
Download [ISPRS](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)

The final folder structure should look like this:

```none
OSDA-ST
├── ...
├── data
│   ├── potsdam
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── vaihingen
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
```
**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for OSDA-SS scenario:

```shell
python tools/convert_datasets/potsdam_full_os.py data/potsdam --nproc 8
python tools/convert_datasets/vaihingen_fill_os.py data/vaihingen --nproc 8
```

### Training
```shell
python run_experiments.py --config configs/daformer/pot2vai_OS_building.py
```

### Testing
```shell
sh test.sh work_dirs/run_name/
```
