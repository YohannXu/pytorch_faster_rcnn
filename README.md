# pytorch_faster_rcnn
Faster RCNN implemented by pytorch

### Requirements

python3.6

torch>=1.4.0

torchvision>=0.5.0

### Usage

训练

```
python train.py
```

验证

```
python val.py
```

测试

```
python infer.py
```

### 训练结果

|   GPU   | 迭代次数 | batch size |  mAP   |
| :-----: | :------: | :--------: | :----: |
| RTX2070 |  90000   |     2      | 26.48% |

