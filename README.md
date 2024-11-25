 Graph Positive-Unlabeled Learning via Bootstrapping Label Disambiguation
====
This repository contains an implementation of a novel method for graph positive and unlabeled learning, 
titled "Graph Positive-Unlabeled Learning via Bootstrapping Label Disambiguation."

##Main ideas
<p align="left">
  <img src="figs/Overview.jpg"/>
<p align="left"><em>Overview of the proposed Bootstrap Label Disambiguation (BLD) framework for graph positive-unlabeled learning. It comprises a node representation learning module based on bootstrapping and a central region-based label disambiguation algorithm, providing useful inputs and precise targets during the training of inner binary classifiers $ g_{\theta}$.  The learning module accepts two augmentations, $ \widetilde{G}_{1} $ and $ \widetilde{G}_{2} $, of a graph $ G $ as input. It then jointly learns both the main network, which comprises a GNN encoder $ f_{\theta} $ and an MLP predictor $ q_{\theta} $, and the auxiliary network, including a GNN encoder $ f_{\phi} $, by conducting two bootstrapped learning tasks on the P set and V set, respectively.  The outputted representation $ \widetilde{Z}_{\theta} $ and positive prototypes $ c_{P} $ help construct a reliable region to facilitate label disambiguation. The disambiguated labels are subsequently used to train the classifier $ g_{\theta} $. Upon convergence, only $ f_{\theta} $ and $ g_{\theta} $ are retained for classifying unlabeled nodes.</em>
</p>

## Requirements
  * numpy==1.24.3
  * scikit_learn==1.3.1
  * torch==2.0.1
  * torch_geometric==2.3.0
## Usage
run BLD demo by:
```python BLD -d 'cora' -c 2 ```

where -d represents the used dataset, and -c denotes the index of label to be used as positive
## Change Log
```
To be updated
```
## References
```
To be updated
```



