# CAPE: Cross-block Adaptive Peer Evaluation for Side-Channel Analysis

Code accompanying the paper **"CAPE: Cross-Block Deep Learning Framework for Non-Profiled Side-Channel Analysis"**

## Overview

CAPE is  the first DL-NSCA framework that successfully exploits cross-block leakage similarity to improve key recovery. This code repository provides an example of applying CAPE to the ASCADv1 fixed-key dataset. CAPE consists of three phases:

**Phase 1:** For the AES algorithm with $B=16$ blocks (each block corresponds to one S-box operation), a neural network is trained for each of the 256 key guesses per block, consistent with standard DL-NSCA with independent block-wise analysis. Unlike existing DL-NSCA methods which end by selecting the minimum-loss key guess as the recovered sub-key, this framework retains the top $N$ key guesses per block, with their corresponding models and hypothetical leakages passed to the next phase. The top $N$ selection is mainly to reduce computational overhead, while $N=256$ can be adopted to retain all key guesses and prevent the correct sub-key from being excluded.

**Phase 2:** Pairwise cross-block UDA training of the models is performed between all blocks. Models from each source block are fine-tuned on the unlabeled traces of every other target block to produce transferred models. For each block, we obtain $B$ model groups, including $1$ native group and $B-1$ transfer-learned groups encoding peer block information. Each group contains $N$ models, corresponding to the top $N$ key guesses of the source block.

**Phase 3:** Each block is scored using its $B$ model groups. For each group, the Pearson correlation between its prediction sequences and the hypothetical leakage sequences of the key guesses of the current block is used as the core metric. We aggregate information from all groups using the proposed CAPE score. The key guess with the highest score is the recovered sub-key.




## Files

| File | Model | Loss |
|------|-------|------|
| `block_sweep_peer0_convwinmcr.py` | CAPE-ConvWinMCR (1D-conv + regression) | MSELoss |
| `block_sweep_peer0_mlpcls.py`     | CAPE-DDLA-MLP (classification)    | CrossEntropyLoss |
| `block_sweep_peer0_cnncls.py`     | CAPE-DDLA-CNN (classification)    | CrossEntropyLoss |

All three scripts share the same three-phase CAPE pipeline; only the model architecture and loss differ.

## Dataset

**ASCADv1** — `ATMega8515_raw_traces.h5`

Download from the [ASCAD repository](https://github.com/ANSSI-FR/ASCAD).  Place the `.h5` file in the **same directory as the script** you intend to run.

Expected HDF5 structure:
```
traces/               float32 array  (N_traces, trace_length)
metadata/
  plaintext           uint8 array    (N_traces, 16)
  key                 uint8 array    (N_traces, 16)
```

## Requirements

```
python >= 3.9
torch >= 1.13
numpy
h5py
scipy
matplotlib
```

## Usage

```bash
# ConvWinMCR regression variant
python block_sweep_peer0_convwinmcr.py

# MLP classification variant
python block_sweep_peer0_mlpcls.py

# CNN classification variant
python block_sweep_peer0_cnncls.py
```

Key parameters to configure at the top of each script:

| Parameter | Description |
|-----------|-------------|
| `GPU_IDS` | List of CUDA device indices to use |
| `NUM_TRACES` | Number of traces to load |
| `TARGET_BLOCK` | Block index (0–13) whose CAPE score is computed |
| `TOP_N` | Number of top-ranked key hypotheses carried into Phase 2 & 3 |
| `INCLUDE_SELF_SCORE` | Whether to add the block's own Pearson diagonal to the peer score |
| `LABEL_MODEL` | `'HW'` (Hamming weight, 9 classes) or `'LSB'` (1 bit, 2 classes) — classification scripts only |

Saved models and figures are written to `models_<variant>_<N>/` and `figures_<variant>_<N>/` next to the script.  Already-trained models are automatically skipped on re-runs.

## Trace Windows

The 14 block windows (sample index ranges) were identified from the ASCADv1 raw traces and correspond to AES key bytes 2–15.  Block 0 (byte 2) is the default `TARGET_BLOCK`.
