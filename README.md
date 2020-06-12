# pytorch_faster_rcnn

Faster RCNN implemented by pytorch

## Requirements

- python3.6
- torch>=1.4.0
- torchvision>=0.5.0

## Install

```bash
git clone https://github.com/YohannXu/pytorch_faster_rcnn.git

cd pytorch_faster_rcnn

# 依赖安装
pip install -r requirements.txt

# apex 安装
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Usage

```bash
# 训练
python train.py

# 验证
python val.py

# 推断（测试）
python infer.py
```

## 训练结果

|   GPU   | 迭代次数  | batch size |  mAP   |
| :-----: | :------: | :--------: | :----: |
| RTX2070 |  90000   |     2      | 26.48% |
