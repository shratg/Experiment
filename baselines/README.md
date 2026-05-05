# Baselines

### [PCBM](https://arxiv.org/abs/2205.15480)

#### 一、概念学习

##### 1. 配置文件路径

在**data/constants.py**中配置各数据集和概念集的地址

##### 2. 运行程序

```
python learn_concepts_dataset.py --dataset-name cub --backbone-name clip:ViT-L/14 --C 0.001 0.01 0.1 1.0 10.0 --n-samples 100 --out-dir conceptbanks --device cuda

dataset-name从以下选择：broden（cifar10 和 cifar100都用这个概念集） awa2 cub
backbone-name从以下选择：clip:ViT-L/14 clip:RN50
```

> 注意这里的dataset-name填的是概念集而非数据集

#### 二、模型训练

```
python train_pcbm.py --concept-bank conceptbanks/awa2_clip_ViT-L_14_10.0_50.pkl --dataset awa2 --backbone-name clip:ViT-L/14 --out-dir conceptbanks --lam 2e-4 --alpha 0.99 --seed 42 --device cuda


dataset从以下选择：cifar100 cifar10 awa2 cub
backbone-name从以下选择：clip:ViT-L/14 clip:RN50
```

> 1. concept-bank选择上一步概念学习后输出的.pkl文件（每个C值都会对应一个）
> 2. 这里的dataset填的是数据集而非概念集
> 3. 训练时的backbone必须和概念学习时的backbone一致，不然会因为向量化维度不同报错
> 4. 数据集cifar10和cifar100默认会这两个数据集会用 torchvision 自动下载到out_dir。如果不成功，可以在**data/data_zoo.py**里改也可以把这俩数据集自己下好放在out_dir文件夹里
> 5. 当前代码的AwA是 50 类全参与，每一类的训练集和测试集比例为8:2 。
