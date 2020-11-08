# ZeroSpeech Challenge 2021 Baseline System

This repository contains the baseline system of the ZeroSpeech Challenge 2021.

The baseline system consists of 3 different componants: Unsupervised Acoustic Modeling (CPC), Clustering (k-means) and Masked Language Modeling (BERT) as described in \cite{paper}.

## Dependencies
In order to run the evaluation and/or train the baseline model. The two following packages are required:
- [CPC_audio (zerospeech branch)](https://github.com/facebookresearch/CPC_audio/tree/zerospeech)
- [fairseq](https://github.com/pytorch/fairseq)

## Checkpoints
The checkpoints can be downloaded from ??? and put in the directory `checkpoints` as follows:
```
checkpoints  
└───CPC-big-kmeans50-BERT
│   └───CPC_big_6kh
│   │   └─checkpoint_32.pt
│   │   └─checkpoint_args.json
│   └───clustering_kmeans50
│   │   └─clustering_CPC_big_kmeans50.pt
│   │   └─clustering_CPC_big_kmeans50_args.json
│   └───BERT
│       └─BERT_CPC_big_kmeans50.pt
│       └─dict.txt
└───...
```

## Evaluations
### Audio quantization
First of all, we need to quantize the audio files with the CPC+clustering modules.

To quantize a set of audio files, run the `clustering_quantization.py` script.

Example command:
```bash
python clustering_quantization.py checkpoints/CPC-big-kmeans50-BERT/clustering_kmeans50/clustering_CPC_big_kmeans50.pt ../data/LibriSpeech/dev-clean/ ../quantized/LibriSpeech/dev-clean/ --file_extension .flac
```

The quantized units will be written in the `quantized_outputs.txt` file in the output directory.

### Build clustering and one-hot features for ABX
We can use the script `build_CPC_features.py` to extract the CPC features from a set of audio files.

Example command:
```bash
python build_CPC_features.py checkpoints/CPC-big-kmeans50-BERT/CPC_big_6kh/checkpoint_32.pt ../data/LibriSpeech/dev-clean/ ../features/CPC/layer2/LibriSpeech/dev-clean/ --file_extension .flac --gru_level 2
```

In order to export one-hot features from a `quantized_outputs.txt` file, use the `build_1hot_features.py` script:

Example command:
```bash
python build_1hot_features.py ../quantized/LibriSpeech/dev-clean/quantized_outputs.txt ../features/onehot/LibriSpeech/dev-clean/
```

The features of each file will be stored in a corresponding file `file_name.npy`.

### Compute pseudo log-probabilities for sWUGGY, sBLIMP
To compute the pseudo log-probabilities of pseudo-units sequences using BERT model, use the `compute_proba_BERT.py` script.

Example command:
```bash
python compute_proba_BERT.py ../quantized/sWUGGY/dev/quantized_outputs.txt ../scores/sWUGGY/dev/pseudo_log_proba.txt checkpoints/CPC-big-kmeans50-BERT/BERT/BERT_CPC_big_kmeans50.pt
```

### Build BERT features for sSIMI
We can export hidden BERT features of pseudo-units sequences with the `build_BERT_features.py` script.

Example command:
```bash
python build_BERT_features.py ../quantized/sSIMI/dev/quantized_outputs.txt ../features/BERT/layer4/sSIMI/dev/ checkpoints/CPC-big-kmeans50-BERT/BERT/BERT_CPC_big_kmeans50.pt --hidden_level 4
```

The features of each file will be stored in a corresponding file `file_name.npy`.

## Training the baseline system
If you want to reproduce the baseline system, please follow the instructions below.

### CPC
To train the CPC model, follow the instructions at https://github.com/facebookresearch/CPC_audio.

Example command:
```bash

```

### K-means
To train the k-means clustering, run the script `clustering_script.py` in the following [repository](https://github.com/facebookresearch/CPC_audio/tree/zerospeech/cpc/criterion/clustering) `CPC_audio/cpc/criterion/clustering/`.

Example command:
```bash

```

### BERT
We train the fairseq's RoBERTa model as similar to this [example](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md), with the exception that we used spans of masks insead of single masks (with additional --mask-multiple-length and --mask-stdev options).

Example command:
```bash

```