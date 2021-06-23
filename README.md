AI CUP Basline
==============

enviroment
----------
- python 3.8
- torch 1.8

Preprocess
----------
- Download pre-trained word embedding: [fasttext](https://fasttext.cc/docs/en/crawl-vectors.html)
- Run ``_preprocess.py`` to generate two file. (``vocab.json`` and ``embeddings.npy``)
- Put your all dataset in ``data`` folder.

Run
---
- Training and testing operations of Risk task are in ``_risk.py``
- Training and testing operations of QA task are in ``_qa.py``

Model
-----
- Hierarchical Attention Networks (see [paper](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf) for more detial)# NLP-task
