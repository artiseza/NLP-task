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
- Download high_risk_words.txt、low_risk_words.txt :(https://drive.google.com/drive/folders/10wx-OX34JZVwhSQtIehMw13iE0BAU364?usp=sharing)
- Put your all dataset、high_risk_words.txt and low_risk_words.txt in ``data`` folder.

Run
---
- Training operations of QA task are in ``_qa.py``
- Testing operations of QA task are in ``predict_qa.py``

Model
-----
- Hierarchical Attention Networks (see [paper](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf) for more detial)# NLP-task
