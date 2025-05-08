# Graph Positive-Unlabeled Learning via Bootstrapping Label Disambiguation (BLD)

## 简介 (Introduction)

本项目是论文 **"Graph Positive-Unlabeled Learning via Bootstrapping Label Disambiguation" (BLD)** 的官方实现。BLD是一个新颖的图学习方法，专为解决图正例-未标记（PU）学习中二元分类的标签模糊性问题而设计。该方法通过其核心的自举式节点表示学习模块与基于中心区域的标签消歧策略，学习出能够对齐正样本的优质节点表示，同时将网络中的模糊未标记节点逐步转化为可靠的正负监督信号。基于这些精确的节点表示和提炼出的监督信息，BLD 能够有效地训练出高性能的二元分类模型，在多个真实图数据集上显著超越现有PU学习方法，甚至达到或优于全监督模型的性能水平。

<!--
**论文链接:**
[Bootstrap Deep Metric for Seed Expansion in Attributed Networks](https://dl.acm.org/doi/10.1145/3626772.3657687)
*Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24)*


## 引用 (Citation)

如果您在研究中发现 BDM 有用，请考虑引用我们的论文：

```bibtex
@inproceedings{10.1145/3626772.3657687,
  author    = {Liang, Chunquan and Wang, Yifan and Chen, Qiankun and Feng, Xinyuan and Wang, Luyue and Li, Mei and Zhang, Hongming},
  title     = {Bootstrap Deep Metric for Seed Expansion in Attributed Networks},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24)},
  year      = {2024},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3626772.3657687},
  pages = {1629–1638},
  numpages = {10},
  location = {Washington DC, USA}, 
}
```
-->

## BDM 框架概览 (Overview of BDM Framework)

BDM 框架的核心思想和主要组件如下图所示：

![image](https://github.com/user-attachments/assets/1344d26e-5bdd-491b-bcb9-48f397c347b2)

*图注：本文概述了针对图正类无标记学习提出的引导标签消歧 (BLD) 框架。该框架包含一个基于引导的节点表示学习模块和一个基于中心区域的标签消歧算法，用于在内部二分类器 $ g_{\theta}$ 的训练过程中提供有用的输入和精确的目标。该学习模块接受图 $ G $ 的两个增强 $ \widetilde{G}_{1} $ 和 $ \widetilde{G}_{2} $ 作为输入。然后，它通过分别在 P 集和 V 集上执行两个引导学习任务，联合学习主网络（包含 GNN 编码器 $ f_{\theta} $ 和 MLP 预测器 $ q_{\theta} $）和辅助网络（包含 GNN 编码器 $ f_{\phi} $）。输出的表征 $ \widetilde{Z}_{\theta} $ 和正样本 $ c_{P} $ 有助于构建一个可靠的区域，从而促进标签消歧。消歧后的标签随后用于训练分类器 $ g_{\theta} $。收敛后，仅保留 $ f_{\theta} $ 和 $ g_{\theta} $ 用于对未标记节点进行分类。*

## 环境要求 (Requirements)

- `numpy==1.24.3`
- `scikit-learn==1.3.1`
- `torch==2.0.1`
- `torch_geometric==2.3.0`

## 使用方法 (Usage)

通过以下命令运行 BDM 演示：

```python
python BLD.py -d 'Cora' -c 0 -s 1 -u 'False'
```

参数说明：
- `-d`: 使用的数据集名称 (e.g., 'Cora', 'CiteSeer', 'PubMed')。
- `-c`: 用作正样本的标签索引。
- `-s`: 随机种子集的大小。
- `-u`: 是否使用高维节点特征作为种子节点 (True/False)。



