"""
block_sweep_peer0_cnncls.py

CAPE on ASCADv1 using a CNN
classification model.  Three-phase pipeline:

  Phase 1 — Train NUM_BLOCKS×256 CNNCLS models (one per block×key guesses).
             Rank hypotheses by final CE loss (lower = more likely correct key).
  Phase 2 — MMD-based unsupervised domain adaptation: adapt each top-N
             source-block model to the target block's trace distribution.
  Phase 3 — Compute CAPE score for the target block via Pearson correlation
             between adapted model predictions and hypothetical leakage values.

Model architecture (CNNCLS, classification):
  Conv1d(1,4, k=32)+ReLU+AvgPool(2)+BN → Conv1d(4,4, k=16)+ReLU+AvgPool(4)+BN
  Size: 700→669→334→319→79  Flatten: 4×79=316
  → Linear(316, FC_SIZE)+ReLU → Linear(FC_SIZE, NUM_CLASSES)
  extract_features: first conv-block output flattened (4×334=1336) for MMD.
  Label modes:
    'HW'  — Hamming weight of SBOX[p⊕k], NUM_CLASSES=9
    'LSB' — LSB of SBOX[p⊕k],            NUM_CLASSES=2
  Loss: CrossEntropyLoss  |  Optimizer: Adam(lr=1e-3)
  predict_hw output:
    'HW'  — softmax-weighted expectation ∑(i·p_i)
    'LSB' — P(LSB=1), i.e. softmax probability of class 1
"""

import os
import copy
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)
HW_TABLE  = np.array([bin(x).count('1') for x in range(256)], dtype=np.uint8)
LSB_TABLE = np.array([x & 1 for x in range(256)], dtype=np.uint8)

# Trace windows for ASCADv1 bytes 2–15 (14 blocks).
# Each tuple is (start, end) sample index within a raw trace.
BLOCK_BYTES = list(range(2, 16))
WINDOWS = [
    (45400, 46100),   # byte 2  → block 0  (target block)
    (32906, 33606),   # byte 3  → block 1
    (47482, 48182),   # byte 4  → block 2
    (41235, 41935),   # byte 5  → block 3
    (37071, 37771),   # byte 6  → block 4
    (34989, 35689),   # byte 7  → block 5
    (26659, 27359),   # byte 8  → block 6
    (39153, 39853),   # byte 9  → block 7
    (28742, 29442),   # byte 10 → block 8
    (43318, 44018),   # byte 11 → block 9
    (20412, 21112),   # byte 12 → block 10
    (22495, 23195),   # byte 13 → block 11
    (49564, 50264),   # byte 14 → block 12
    (18330, 19030),   # byte 15 → block 13
]
NUM_BLOCKS   = len(BLOCK_BYTES)   # 14
TARGET_BLOCK = 0                  # index of the block for which CAPE score is computed

# CNN architecture constants (all derived from SEG_LEN; do not change independently)
SEG_LEN       = WINDOWS[0][1] - WINDOWS[0][0]   # 700
CONV1_FILTERS = 4
CONV1_KERNEL  = 32
POOL1_SIZE    = 2
CONV2_FILTERS = 4
CONV2_KERNEL  = 16
POOL2_SIZE    = 4
# Size derivation (stride=1, no padding):
#   Conv1: 700-32+1=669  AvgPool(2): 669//2=334
#   Conv2: 334-16+1=319  AvgPool(4): 319//4=79
#   Flatten: 4*79=316
_after_conv1 = SEG_LEN - CONV1_KERNEL + 1                  # 669
_after_pool1 = _after_conv1 // POOL1_SIZE                   # 334
_after_conv2 = _after_pool1 - CONV2_KERNEL + 1              # 319
_after_pool2 = _after_conv2 // POOL2_SIZE                   # 79
FLAT_SIZE    = CONV2_FILTERS * _after_pool2                 # 316  (FC input)
FEAT_SIZE    = CONV1_FILTERS * _after_pool1                 # 1336 (MMD feature dim)
FC_SIZE      = 128

# Training hyperparameters
SRC_EPOCHS  = 100
ADA_EPOCHS  = 50
BATCH_SIZE  = 200
LR_SRC      = 1e-3
LR_ADA      = 2e-4
TRAIN_RATIO = 1.0
MMD_LAMBDA  = 1000.0
MMD_BANDWIDTHS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Configurable run parameters
GPU_IDS            = [0, 1, 2, 3, 4, 5, 6, 7]
TOP_N              = 256
NUM_TRACES         = 10000
INCLUDE_SELF_SCORE = False   # if True, add the block's own Pearson diagonal to the peer score
LABEL_MODEL        = 'HW'   # 'HW': Hamming weight (9 classes); 'LSB': LSB (2 classes)

assert LABEL_MODEL in ('HW', 'LSB'), "LABEL_MODEL must be 'HW' or 'LSB'"
NUM_CLASSES = 9 if LABEL_MODEL == 'HW' else 2

TRUE_KEYS = None   # populated in main() from the dataset

# Output directories — resolved relative to this script's location
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_BASE_DIR, f'models_cnncls_{LABEL_MODEL.lower()}_{NUM_TRACES}')
FIG_DIR   = os.path.join(_BASE_DIR, f'figures_cnncls_{LABEL_MODEL.lower()}_{NUM_TRACES}')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)

H5_PATH = os.path.join(_BASE_DIR, 'ATMega8515_raw_traces.h5')

HW_WEIGHTS = torch.arange(NUM_CLASSES, dtype=torch.float32)   # [0..8] for HW, [0,1] for LSB


class CNNCLS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, CONV1_FILTERS, kernel_size=CONV1_KERNEL, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=POOL1_SIZE),
            nn.BatchNorm1d(CONV1_FILTERS),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=CONV2_KERNEL, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=POOL2_SIZE),
            nn.BatchNorm1d(CONV2_FILTERS),
        )
        self.fc   = nn.Sequential(nn.Linear(FLAT_SIZE, FC_SIZE), nn.ReLU())
        self.head = nn.Linear(FC_SIZE, NUM_CLASSES)

    def forward(self, x):
        # x: (B, SEG_LEN) → (B, 1, SEG_LEN)
        x = x.unsqueeze(1)
        x = self.conv2(self.conv1(x))   # (B, CONV2_FILTERS, _after_pool2)
        x = x.flatten(1)                # (B, FLAT_SIZE)
        x = self.fc(x)
        return self.head(x)             # (B, NUM_CLASSES) — logits

    def extract_features(self, x):
        """Return first conv-block output flattened (FEAT_SIZE dim) for MMD computation."""
        feat = self.conv1(x.unsqueeze(1))   # (B, CONV1_FILTERS, _after_pool1)
        return feat.flatten(1)              # (B, FEAT_SIZE)


def rbf_kernel(X, Y, bw):
    XX = (X * X).sum(1, keepdim=True)
    YY = (Y * Y).sum(1, keepdim=True).t()
    return torch.exp(-(XX + YY - 2.0 * X @ Y.t()) / (2.0 * bw ** 2))

def mmd_loss(sf, tf):
    loss = torch.zeros(1, device=sf.device)
    for bw in MMD_BANDWIDTHS:
        loss += rbf_kernel(sf, sf, bw).mean() + rbf_kernel(tf, tf, bw).mean() \
                - 2.0 * rbf_kernel(sf, tf, bw).mean()
    return loss / len(MMD_BANDWIDTHS)


def seg_traces(traces, block_idx):
    win_l, win_r = WINDOWS[block_idx]
    return traces[:, win_l:win_r].astype(np.float32)

def label_hyp(plain, key_byte, block_idx):
    """Return int64 classification labels: HW ∈ {0..8} or LSB ∈ {0,1}."""
    byte_col = BLOCK_BYTES[block_idx]
    sbox_out = SBOX[plain[:, byte_col] ^ np.full(len(plain), key_byte, dtype=np.uint8)]
    if LABEL_MODEL == 'HW':
        return HW_TABLE[sbox_out].astype(np.int64)
    else:
        return LSB_TABLE[sbox_out].astype(np.int64)

def label_hyp_float(plain, key_byte, block_idx):
    """Return float64 labels for Pearson correlation scoring."""
    return label_hyp(plain, key_byte, block_idx).astype(np.float64)

def make_loader(x, y, shuffle, device_bs=None):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    bs = min(device_bs or BATCH_SIZE, len(ds))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, drop_last=True)

def make_loader_unlabeled(x, shuffle, device_bs=None):
    dummy = np.zeros(len(x), dtype=np.int64)
    return make_loader(x, dummy, shuffle, device_bs)

def eval_ce(model, x_np, y_np, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_np).to(device))
        labels = torch.from_numpy(y_np).to(device)
        ce     = criterion(logits, labels).item()
    return ce

def predict_hw(model, x_np, device):
    """Return continuous predictions (float64) for Pearson scoring.

    HW mode:  softmax-weighted expectation ∑(i·p_i)
    LSB mode: P(LSB=1), i.e. softmax probability of class 1
    """
    model.eval()
    weights = HW_WEIGHTS.to(device)
    out     = []
    loader  = DataLoader(torch.from_numpy(x_np), batch_size=1024, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            probs = torch.softmax(model(xb.to(device)), dim=1)
            val   = (probs * weights).sum(dim=1) if LABEL_MODEL == 'HW' else probs[:, 1]
            out.append(val.cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _to_shared_memory(arr: np.ndarray):
    shm = SharedMemory(create=True, size=arr.nbytes)
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[:] = arr
    return shm, shm.name, arr.shape, arr.dtype


def _p1_worker(task_queue, result_queue, gpu_id,
               shm_traces_name, traces_shape, traces_dtype,
               shm_plain_name,  plain_shape,  plain_dtype,
               block_stats, tr_idx, model_dir):
    shm_tr = SharedMemory(name=shm_traces_name)
    traces_shared = np.ndarray(traces_shape, dtype=traces_dtype, buffer=shm_tr.buf)
    shm_pl = SharedMemory(name=shm_plain_name)
    plain_shared  = np.ndarray(plain_shape,  dtype=plain_dtype,  buffer=shm_pl.buf)

    device = torch.device(f'cuda:{gpu_id}')
    pid    = os.getpid()

    while True:
        try:
            item = task_queue.get(timeout=5)
        except Exception:
            break
        if item is None:
            break

        b, k = item
        model_path = os.path.join(model_dir, f'block{b}', f'setkey{k}.pt')
        curve_path = os.path.join(model_dir, f'block{b}', f'setkey{k}_loss.npy')

        if os.path.exists(model_path) and os.path.exists(curve_path):
            curve = np.load(curve_path)
            result_queue.put((b, k, float(curve[-1]), curve))
            continue

        mu, sigma  = block_stats[b]
        x_norm     = (seg_traces(traces_shared, b) - mu) / (sigma + 1e-8)
        y          = label_hyp(plain_shared, k, b)
        x_tr, y_tr = x_norm[tr_idx], y[tr_idx]

        model     = CNNCLS().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SRC)
        loader    = make_loader(x_tr, y_tr, shuffle=True)

        curve = np.zeros(SRC_EPOCHS, dtype=np.float32)

        for ep in range(SRC_EPOCHS):
            model.train()
            t_loss = t_total = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                t_loss  += loss.item() * xb.size(0)
                t_total += xb.size(0)
            curve[ep] = t_loss / t_total

        torch.save(model.state_dict(), model_path)
        np.save(curve_path, curve)

        result_queue.put((b, k, float(curve[-1]), curve))
        print(f'  [GPU{gpu_id} PID{pid}] Block{b}(byte{BLOCK_BYTES[b]}) key={k:3d} done  '
              f'final_ce={curve[-1]:.4f}', flush=True)


def _p2_worker(task_queue, done_queue, gpu_id,
               shm_traces_name, traces_shape, traces_dtype,
               shm_plain_name,  plain_shape,  plain_dtype,
               block_stats, tr_idx, true_keys, model_dir):
    shm_tr = SharedMemory(name=shm_traces_name)
    traces_shared = np.ndarray(traces_shape, dtype=traces_dtype, buffer=shm_tr.buf)
    shm_pl = SharedMemory(name=shm_plain_name)
    plain_shared  = np.ndarray(plain_shape,  dtype=plain_dtype,  buffer=shm_pl.buf)

    device = torch.device(f'cuda:{gpu_id}')
    pid    = os.getpid()

    while True:
        try:
            item = task_queue.get(timeout=5)
        except Exception:
            break
        if item is None:
            break

        i, rank, k = item
        j = TARGET_BLOCK
        tgt_dir  = os.path.join(model_dir, f'block{i}_to_block{j}')
        os.makedirs(tgt_dir, exist_ok=True)
        ada_path = os.path.join(tgt_dir, f'rank{rank}_key{k}.pt')

        if os.path.exists(ada_path):
            done_queue.put((i, rank, k))
            print(f'  [GPU{gpu_id}] Block{i}→Block{j} rank{rank} key{k} skip', flush=True)
            continue

        src_path  = os.path.join(model_dir, f'block{i}', f'setkey{k}.pt')
        src_model = CNNCLS().to(device)
        src_model.load_state_dict(torch.load(src_path, map_location=device))
        src_model.eval()

        mu_i, sigma_i = block_stats[i]
        x_i_norm = (seg_traces(traces_shared, i) - mu_i) / (sigma_i + 1e-8)
        y_i      = label_hyp(plain_shared, k, i)
        x_src_tr = x_i_norm[tr_idx]
        y_src_tr = y_i[tr_idx]

        mu_j, sigma_j = block_stats[j]
        x_j_norm = (seg_traces(traces_shared, j) - mu_j) / (sigma_j + 1e-8)
        # Use the true key to evaluate CE on the target block for best-state selection.
        y_j_eval = label_hyp(plain_shared, true_keys[j], j)

        model      = copy.deepcopy(src_model).to(device)
        criterion  = nn.CrossEntropyLoss()
        optimizer  = torch.optim.Adam(model.parameters(), lr=LR_ADA)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADA_EPOCHS)
        src_loader = make_loader(x_src_tr, y_src_tr, shuffle=True)
        tgt_loader = make_loader_unlabeled(x_j_norm, shuffle=True)

        best_ce    = eval_ce(src_model, x_j_norm, y_j_eval, device)
        best_state = copy.deepcopy(model.state_dict())

        for _ in range(ADA_EPOCHS):
            model.train()
            for (xs, ys), (xt, _) in zip(src_loader, tgt_loader):
                xs, ys = xs.to(device), ys.to(device)
                xt     = xt.to(device)
                optimizer.zero_grad()
                ce   = criterion(model(xs), ys)
                loss = ce + MMD_LAMBDA * mmd_loss(model.extract_features(xs),
                                                   model.extract_features(xt))
                loss.backward()
                optimizer.step()
            scheduler.step()
            ce_val = eval_ce(model, x_j_norm, y_j_eval, device)
            if ce_val < best_ce:
                best_ce    = ce_val
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        torch.save(model.state_dict(), ada_path)
        done_queue.put((i, rank, k))
        print(f'  [GPU{gpu_id} PID{pid}] Block{i}→Block{j} rank{rank} key{k} done', flush=True)


def phase1_parallel(traces, plain, true_keys, rng):
    block_stats = []
    for b in range(NUM_BLOCKS):
        x_raw = seg_traces(traces, b)
        block_stats.append((x_raw.mean(), x_raw.std()))

    N      = traces.shape[0]
    tr_idx = rng.permutation(N)[:int(N * TRAIN_RATIO)]

    for b in range(NUM_BLOCKS):
        os.makedirs(os.path.join(MODEL_DIR, f'block{b}'), exist_ok=True)

    print('  [Phase1] Copying data to shared memory...', flush=True)
    shm_tr, shm_tr_name, tr_shape, tr_dtype = _to_shared_memory(traces)
    shm_pl, shm_pl_name, pl_shape, pl_dtype = _to_shared_memory(plain)
    print('  [Phase1] Shared memory ready.', flush=True)

    task_queue   = mp.Queue()
    result_queue = mp.Queue()
    total_tasks  = NUM_BLOCKS * 256

    for b in range(NUM_BLOCKS):
        for k in range(256):
            task_queue.put((b, k))
    for _ in GPU_IDS:
        task_queue.put(None)

    workers = []
    for gpu_id in GPU_IDS:
        p = mp.Process(
            target=_p1_worker,
            args=(task_queue, result_queue, gpu_id,
                  shm_tr_name, tr_shape, tr_dtype,
                  shm_pl_name, pl_shape, pl_dtype,
                  block_stats, tr_idx, MODEL_DIR),
            daemon=True
        )
        p.start()
        workers.append(p)

    final_losses   = [[None]*256 for _ in range(NUM_BLOCKS)]
    loss_curves    = [[None]*256 for _ in range(NUM_BLOCKS)]
    block0_done    = 0
    block0_plotted = False
    done = 0
    while done < total_tasks:
        b, k, fl, curve = result_queue.get()
        final_losses[b][k] = fl
        loss_curves[b][k]  = curve
        done += 1
        if b == TARGET_BLOCK:
            block0_done += 1
            if block0_done == 256 and not block0_plotted:
                curves_for_plot = [(k, loss_curves[TARGET_BLOCK][k]) for k in range(256)]
                _plot_loss_curves(TARGET_BLOCK, curves_for_plot, true_keys[TARGET_BLOCK])
                block0_plotted = True
        if done % 100 == 0 or done == total_tasks:
            print(f'  [Phase1] {done}/{total_tasks} tasks done', flush=True)

    for p in workers:
        p.join()

    shm_tr.close(); shm_tr.unlink()
    shm_pl.close(); shm_pl.unlink()

    return final_losses, loss_curves, block_stats, tr_idx


def _plot_loss_curves(block_idx, curves_for_plot, true_key):
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, curve in curves_for_plot:
        if k != true_key:
            ax.plot(curve, color='gray', linewidth=0.4, alpha=0.4)
    for k, curve in curves_for_plot:
        if k == true_key:
            ax.plot(curve, color='red', linewidth=1.2, label=f'true key={true_key}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss (CE)')
    ax.set_title(f'Block {block_idx} (byte {BLOCK_BYTES[block_idx]}) — Phase 1 CE Loss (all 256 keys)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f'phase1_loss_block{block_idx}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  [Plot] Saved {path}')


def phase2_to_target(traces, plain, final_losses, block_stats, tr_idx, true_keys):
    top_keys = []
    for b in range(NUM_BLOCKS):
        sorted_keys = sorted(range(256), key=lambda k: final_losses[b][k])
        top_keys.append(sorted_keys[:TOP_N])
        print(f'  [Block{b}(byte{BLOCK_BYTES[b]})] top-{TOP_N} keys: {top_keys[b]}')

    task_queue  = mp.Queue()
    done_queue  = mp.Queue()
    src_blocks  = [i for i in range(NUM_BLOCKS) if i != TARGET_BLOCK]
    total_tasks = len(src_blocks) * TOP_N

    for i in src_blocks:
        for rank, k in enumerate(top_keys[i]):
            task_queue.put((i, rank, k))
    for _ in GPU_IDS:
        task_queue.put(None)

    print('  [Phase2] Copying data to shared memory...', flush=True)
    shm_tr, shm_tr_name, tr_shape, tr_dtype = _to_shared_memory(traces)
    shm_pl, shm_pl_name, pl_shape, pl_dtype = _to_shared_memory(plain)
    print('  [Phase2] Shared memory ready.', flush=True)

    workers = []
    for gpu_id in GPU_IDS:
        p = mp.Process(
            target=_p2_worker,
            args=(task_queue, done_queue, gpu_id,
                  shm_tr_name, tr_shape, tr_dtype,
                  shm_pl_name, pl_shape, pl_dtype,
                  block_stats, tr_idx, true_keys, MODEL_DIR),
            daemon=True
        )
        p.start()
        workers.append(p)

    done = 0
    while done < total_tasks:
        done_queue.get()
        done += 1
        if done % 50 == 0 or done == total_tasks:
            print(f'  [Phase2] {done}/{total_tasks} tasks done', flush=True)

    for p in workers:
        p.join()

    shm_tr.close(); shm_tr.unlink()
    shm_pl.close(); shm_pl.unlink()

    return top_keys


def _pearson_col_max(preds, hyp_hws_j):
    """Build a (TOP_N×TOP_N) Pearson matrix and return its column-wise maximum.

    Uses matrix multiplication to compute all pairwise correlations at once,
    replacing an O(TOP_N^2) Python loop.
    """
    P = np.stack(preds,     axis=0)   # (TOP_N, N)
    H = np.stack(hyp_hws_j, axis=0)   # (TOP_N, N)

    P = P - P.mean(axis=1, keepdims=True)
    H = H - H.mean(axis=1, keepdims=True)

    cov   = P @ H.T                                       # (TOP_N, TOP_N)
    denom = np.outer(np.linalg.norm(P, axis=1),
                     np.linalg.norm(H, axis=1)) + 1e-12
    mat = cov / denom
    mat = np.where(np.isfinite(mat), mat, 0.0)
    return np.max(mat, axis=0)


def phase3_peer_score_target(traces, plain, top_keys, block_stats):
    infer_device = torch.device(f'cuda:{GPU_IDS[0]}')
    j = TARGET_BLOCK

    mu_j, sigma_j = block_stats[j]
    x_j_norm  = (seg_traces(traces, j) - mu_j) / (sigma_j + 1e-8)
    hyp_hws_j = [label_hyp_float(plain, k_j, j) for k_j in top_keys[j]]

    peer_score = np.zeros(TOP_N, dtype=np.float64)
    self_score = np.zeros(TOP_N, dtype=np.float64)

    src_blocks = [i for i in range(NUM_BLOCKS) if i != j]
    for i in src_blocks:
        preds = []
        for rank, k in enumerate(top_keys[i]):
            model_path = os.path.join(MODEL_DIR, f'block{i}_to_block{j}',
                                      f'rank{rank}_key{k}.pt')
            model = CNNCLS().to(infer_device)
            model.load_state_dict(torch.load(model_path, map_location=infer_device))
            preds.append(predict_hw(model, x_j_norm, infer_device))

        col_max = _pearson_col_max(preds, hyp_hws_j)
        peer_score += col_max
        print(f'  [Pearson ({i}→{j})] col_max_max={col_max.max():.4f}', flush=True)

    if INCLUDE_SELF_SCORE:
        preds_self = []
        for rank, k in enumerate(top_keys[j]):
            model_path = os.path.join(MODEL_DIR, f'block{j}', f'setkey{k}.pt')
            model = CNNCLS().to(infer_device)
            model.load_state_dict(torch.load(model_path, map_location=infer_device))
            preds_self.append(predict_hw(model, x_j_norm, infer_device))

        P  = np.stack(preds_self, axis=0)   # (TOP_N, N)
        H  = np.stack(hyp_hws_j,  axis=0)
        Pc = P - P.mean(axis=1, keepdims=True)
        Hc = H - H.mean(axis=1, keepdims=True)
        numer = (Pc * Hc).sum(axis=1)
        denom = np.linalg.norm(Pc, axis=1) * np.linalg.norm(Hc, axis=1) + 1e-12
        # Rank-r model was trained with top_keys[j][r], so the Pearson matrix
        # diagonal directly gives the one-to-one self-correlation without col-max.
        diag       = numer / denom
        self_score = np.where(np.isfinite(diag), diag, 0.0)
        print(f'  [Pearson ({j}→{j}, self-diag)] diag_max={self_score.max():.4f}', flush=True)

    total_score = peer_score + self_score
    return total_score, peer_score, self_score


def plot_peer_score(total_score, peer_score, self_score, top_keys_target, true_key):
    keys         = top_keys_target
    colors       = ['red' if k == true_key else 'steelblue' for k in keys]
    score_label  = 'Total Score (peer + self)' if INCLUDE_SELF_SCORE else 'Peer Score'
    title_suffix = ' [+self]' if INCLUDE_SELF_SCORE else ''
    fname        = f'peer_score_block{TARGET_BLOCK}{"_with_self" if INCLUDE_SELF_SCORE else ""}.png'

    fig, ax = plt.subplots(figsize=(max(4, TOP_N * 0.8), 4))
    ax.bar(range(TOP_N), total_score, color=colors, width=0.6)
    ax.set_xticks(range(TOP_N))
    ax.set_xticklabels([f'{k}\n(r{r})' for r, k in enumerate(keys)], fontsize=7)
    ax.set_xlabel('Key guesses (rank by Phase-1 CE loss)')
    ax.set_ylabel(score_label)
    ax.set_title(f'Peer Score{title_suffix} — Block {TARGET_BLOCK} (byte {BLOCK_BYTES[TARGET_BLOCK]})')
    ax.legend(handles=[
        Patch(facecolor='red',       label=f'true key={true_key}'),
        Patch(facecolor='steelblue', label='other'),
    ], fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  [Plot] Saved {path}')


def main():
    global TRUE_KEYS

    print(f'Loading data from {H5_PATH} ...', flush=True)
    with h5py.File(H5_PATH, 'r') as f:
        all_traces = f['traces'][:NUM_TRACES].astype(np.float32)
        meta       = f['metadata'][:NUM_TRACES]
        plain      = meta['plaintext'].astype(np.uint8)
        key_all    = meta['key'].astype(np.uint8)

    TRUE_KEYS = [int(key_all[0, b]) for b in BLOCK_BYTES]

    print(f'Loaded traces={all_traces.shape}  plain={plain.shape}')
    print(f'GPU_IDS={GPU_IDS}  ({len(GPU_IDS)} GPUs)')
    print(f'NUM_BLOCKS={NUM_BLOCKS}  TARGET_BLOCK={TARGET_BLOCK} '
          f'(byte {BLOCK_BYTES[TARGET_BLOCK]})  TOP_N={TOP_N}  NUM_TRACES={NUM_TRACES}')
    print(f'Model: CNNCLS  LABEL_MODEL={LABEL_MODEL}  SEG_LEN={SEG_LEN}  '
          f'FLAT_SIZE={FLAT_SIZE}  FEAT_SIZE={FEAT_SIZE}  NUM_CLASSES={NUM_CLASSES}')
    print(f'TRUE_KEYS={TRUE_KEYS}\n')

    rng = np.random.default_rng(42)

    print('=' * 70)
    print(f'[Phase 1] Training {NUM_BLOCKS}×256 models (skip existing)...')
    print('=' * 70)
    final_losses, loss_curves, block_stats, tr_idx = phase1_parallel(
        all_traces, plain, TRUE_KEYS, rng
    )

    for b in range(NUM_BLOCKS):
        top       = sorted(range(256), key=lambda k: final_losses[b][k])[:TOP_N]
        rank_true = top.index(TRUE_KEYS[b]) + 1 if TRUE_KEYS[b] in top else f'>{TOP_N}'
        print(f'  Block{b:2d}(byte{BLOCK_BYTES[b]:2d})  true_key={TRUE_KEYS[b]}  rank={rank_true}')

    src_blocks = [i for i in range(NUM_BLOCKS) if i != TARGET_BLOCK]
    print('\n' + '=' * 70)
    print(f'[Phase 2] UDA: blocks {src_blocks} → block{TARGET_BLOCK} only'
          f'  ({len(src_blocks)}×{TOP_N} tasks)')
    print('=' * 70)
    top_keys = phase2_to_target(
        all_traces, plain, final_losses, block_stats, tr_idx, TRUE_KEYS
    )

    print('\n' + '=' * 70)
    print(f'[Phase 3] Pearson + Peer Score for block{TARGET_BLOCK}...')
    print('=' * 70)
    total_score, peer_score, self_score = phase3_peer_score_target(
        all_traces, plain, top_keys, block_stats
    )

    plot_peer_score(total_score, peer_score, self_score,
                    top_keys[TARGET_BLOCK], TRUE_KEYS[TARGET_BLOCK])

    score_tag = 'Total Score (peer+self)' if INCLUDE_SELF_SCORE else 'Peer Score'
    print('\n' + '=' * 70)
    print(f'[Summary] Block{TARGET_BLOCK} (byte {BLOCK_BYTES[TARGET_BLOCK]})  '
          f'true_key={TRUE_KEYS[TARGET_BLOCK]}  TOP_N={TOP_N}  NUM_TRACES={NUM_TRACES}  '
          f'INCLUDE_SELF_SCORE={INCLUDE_SELF_SCORE}')
    print('=' * 70)
    print(f'\n[{score_tag} — Block{TARGET_BLOCK}]')
    for r, k in enumerate(top_keys[TARGET_BLOCK]):
        marker   = ' ← TRUE KEY' if k == TRUE_KEYS[TARGET_BLOCK] else ''
        self_str = f'  self={self_score[r]:.4f}' if INCLUDE_SELF_SCORE else ''
        print(f'  rank{r:3d}  key={k:3d}  peer={peer_score[r]:.4f}{self_str}  '
              f'total={total_score[r]:.4f}{marker}')

    true_key = TRUE_KEYS[TARGET_BLOCK]
    if true_key in top_keys[TARGET_BLOCK]:
        orig_rank = top_keys[TARGET_BLOCK].index(true_key) + 1
    else:
        all_sorted = sorted(range(256), key=lambda k: final_losses[TARGET_BLOCK][k])
        orig_rank  = all_sorted.index(true_key) + 1

    score_order    = np.argsort(total_score)[::-1].tolist()
    score_rank_idx = (score_order.index(top_keys[TARGET_BLOCK].index(true_key)) + 1
                      if true_key in top_keys[TARGET_BLOCK] else f'>{TOP_N}')

    print('\n' + '=' * 70)
    print(f'[Key Ranking Summary]  Block{TARGET_BLOCK} (byte {BLOCK_BYTES[TARGET_BLOCK]})  '
          f'true_key={true_key}')
    print(f'  Phase-1 CE loss rank (original) : {orig_rank} / 256')
    print(f'  {score_tag} rank         : {score_rank_idx} / {TOP_N}')
    print('=' * 70)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
