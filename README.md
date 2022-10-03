# UGCL
The open source code for ICDM2022 paper "Unifying Graph Contrastive Learning with Flexible Contextual Scopes"

![overall](https://user-images.githubusercontent.com/75228223/193508189-13fd8fa3-478c-4c64-af46-5e15945d075a.png)

# Overview
Our implementation for UGCL (Unfiying Graph Contrastive Learning) is based on PyTorch.

**To run on cora/citeseer/pubmed**

```
cd train_cora_citeseer_pubmed
python train_gp.py --data 'cora' --epoch 250 --patience 50 --lr 0.0002 --n 9 --hidden 4196 --sample_size 1500 #for cora
python train_gp.py --data 'citeseer' --epoch 150 --patience 50 --lr 5e-05 --n 2 --hidden 8192 --sample_size 3000 #for citeseer
python train_gp.py --data 'pubmed' --epoch 300 --patience 50 --lr 0.0002 --n 14 --sample_size 7000 --hidden 8192 #for pubmed
```

**To run on ogbn-arxiv**
```
cd train_ogbn
python train_ogbn --data 'ogbn' --n 4 --sample_size 2000 --hidden 8192 --test_epoch 2000 --patience 1000
```
