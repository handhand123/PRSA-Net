## Pyramid Region-based Slot Attention Network for Temporal Action Proposal Generation

-[Paper][paper_arixv]

# Usage Guide

## Prerequisites


The training and testing in PRSA-Net is reimplemented in PyTorch for the ease of use. 

- [PyTorch 1.8.1][pytorch]
                   

Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
```

 

## Code and Data Preparation


### Download Datasets

We support experimenting with two publicly available datasets for 
temporal action detection: THUMOS14 & ActivityNet v1.3. Here are some steps to download these two datasets.

- THUMOS14:  [THUMOS14 challenge website][thumos14].
- ActivityNet v1.3: Using the [official ActivityNet downloader][anet_down] to download videos from the YouTube. And the dataset is provided in the form of YouTube URL list. 

### Download Features

You can get the TSN features for training and testing from  [G-TAD][G-TAD] [GoogleDrive][TSN].
I3D features is provided in PGCN[PGCN_github]

## Training


Install Align1D layers
```bash
cd aligner/
python setup.py install
```

Set the path of features in config/cfg.yaml

```bash
feature_path: $PATH_OF_FEATURES
video_info_path: $PATH_OF_ANNOTATIONS
```


Then, you can use the following commands to train PRSA-Net

```bash
python main.py --mode train --cfg $PATH_TO_CONFIG_FILE
```

## Testing Trained Models


You can evaluate the model's action proposal generation performance and action detection performance at the same time by running the following command

```bash
python main.py --mode infer --cfg $PATH_TO_CONFIG_FILE
```

We use the weight of the 4-th epoch by default for model evaluation during the experiment. If you want, you can modify the eval_model field in the config file.
This script will report the proposal generation performance in terms of AR(average recall) under various number of proposals, and detection performance in terms of (mean average precision) at different IoU thresholds..

### proposal generation performance on THUMOS14

| AR@100         | RGB+Flow |
| -------------- | -------- |
| PRSA-Net (I3D) | 56.12    |

### detection performance on THUMOS14

| mAP@0.5IoU (%)              | RGB+Flow |
| --------------------------- | -------- |
| PRSA-Net (I3D + two-stream) | 55.0     |
| PRSA-Net (I3D + PGCN)       | 58.7     |



## Reference
My implementations borrow ideas from previous works.

[BMN: BMN: Boundary-Matching Network for Temporal Action Proposal Generation.][BMN]

[G-TAD: Sub-Graph Localization for Temporal Action Detection][G-TAD]


## Contact

```
lishuaicheng@sensetime.com
```

[pytorch]:https://pytorch.org/
[thumos14]:http://crcv.ucf.edu/THUMOS14/download.html
[anet_down]:https://github.com/activitynet/ActivityNet/tree/master/Crawler
[PGCN_github]: https://github.com/Alvin-Zeng/PGCN
[BMN]: https://arxiv.org/pdf/1907.09702.pdf
[G-TAD]: https://arxiv.org/pdf/1911.11462.pdf
[TSN]: https://drive.google.com/drive/folders/10PGPMJ9JaTZ18uakPgl58nu7yuKo8M_k
[paper_arixv]: https://arxiv.org/pdf/2206.10095.pdf