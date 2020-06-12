# pytorch_faster_rcnn

Faster RCNN implemented by pytorch.

## 1 Requirements

- python3.6
- torch>=1.4.0
- torchvision>=0.5.0

## 2 Install

### 2.1 直接安装

```bash
# 1 下载项目
git clone https://github.com/YohannXu/pytorch_faster_rcnn.git
cd pytorch_faster_rcnn

# 2 依赖安装
pip install -r requirements.txt

# 3 apex 安装
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### 2.2 docker 安装

```bash
# 1 下载项目
git clone https://github.com/YohannXu/pytorch_faster_rcnn.git
cd pytorch_faster_rcnn

# 2 构建镜像
docker image build -t pytorch_faster_rcnn .

# 3 运行实例
docker run -it pytorch_faster_rcnn /bin/bash
```

## 3 Usage

```bash
# 训练
python train.py

# 验证
python val.py

# 推断（测试）
python infer.py
```

## 4 训练结果

|   GPU   | 迭代次数  | batch size |  mAP   |
| :-----: | :------: | :--------: | :----: |
| RTX2070 |  90000   |     2      | 26.48% |

## 5 TODO

- [ ] 直接安装未测试
- [ ] docker 安装未测试
