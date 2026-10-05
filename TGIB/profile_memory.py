"""Memory/row profiler for TGIB. Run this before/after changing the tem_conv recursion.

Reproduces learn_edge.py's data prep exactly, then reports how many rows enter
attention for each (bs, two_hop_depth, row_dedup) configuration. The peak-row
column converts to the size of the largest single fp16 projection, which is
the tensor that OOMs.

    python profile_memory.py                      # cpu, fast (attention stubbed)
    python profile_memory.py --real               # cpu, slow: real attention math
    python profile_memory.py --dedup-detail       # per-call dedup compression

Attention math is stubbed by default because row counts depend only on the
tem_conv recursion shape, not on the attention arithmetic -- this makes the
default run finish in seconds. --real keeps the math and can then confirm there
is no numerical divergence.

Reads processed/ml_<data>.{csv,npy} and processed/ml_<data>_node.npy, i.e. the
same files learn_edge.py expects. Nothing here is written to disk.
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch

from module import TGIB, AttnModel
from graph import NeighborFinder

p = argparse.ArgumentParser()
p.add_argument('--data', type=str, default='wikipedia')
p.add_argument('-d', '--n_degree', type=int, default=20)
p.add_argument('--n_layer', type=int, default=2)
p.add_argument('--n_head', type=int, default=2)
p.add_argument('--bs', type=int, nargs='+', default=[10, 20])
p.add_argument('--steps', type=int, default=3, help='smoke steps when --real')
p.add_argument('--real', action='store_true', help='do not stub attention math')
p.add_argument('--dedup-detail', action='store_true', help='per-call dedup table')
p.add_argument('--uniform', action='store_true')
args = p.parse_args()

g_df = pd.read_csv(f'./processed/ml_{args.data}.csv')
e_feat = np.load(f'./processed/ml_{args.data}.npy')
n_feat = np.load(f'./processed/ml_{args.data}_node.npy')

val_time, _ = list(np.quantile(g_df.ts, [0.70, 0.85]))
src_l, dst_l = g_df.u.values, g_df.i.values
e_idx_l, ts_l = g_df.idx.values, g_df.ts.values
max_idx = max(src_l.max(), dst_l.max())

random.seed(2020)
total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
mask_node_set = set(random.sample(
    list(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))),
    int(0.1 * len(total_node_set))))
none_flag = (1 - g_df.u.map(lambda x: x in mask_node_set).values) * \
            (1 - g_df.i.map(lambda x: x in mask_node_set).values)
train_flag = (ts_l <= val_time) * (none_flag > 0)

adj_list = [[] for _ in range(max_idx + 1)]
for s, d, e, t in zip(src_l[train_flag], dst_l[train_flag],
                      e_idx_l[train_flag], ts_l[train_flag]):
    adj_list[s].append((d, e, t))
    adj_list[d].append((s, e, t))

MODEL_DIM = 3 * n_feat.shape[1]
K = args.n_degree
node_feats_dead = np.count_nonzero(n_feat) == 0

print(f'data={args.data}  train_edges={int(train_flag.sum()):,}  nodes={max_idx + 1:,}  '
      f'feat_dim={n_feat.shape[1]}  model_dim={MODEL_DIM}')
print(f'K={K}  num_layers={args.n_layer}  '
      f'node features all zero: {node_feats_dead}')

torch.manual_seed(2020)
model = TGIB(NeighborFinder(adj_list, uniform=args.uniform), n_feat, e_feat, 64,
             num_layers=args.n_layer, use_time='time', agg_method='attn',
             attn_mode='prod', n_head=args.n_head, drop_out=0.1)

rows_seen = []
if args.real:
    # keep the real math, just count the rows that enter it
    _orig_attn_fwd = AttnModel.forward

    def counting(self, src, src_t, seq, seq_t, seq_e, mask):
        rows_seen.append(int(src.shape[0]))
        return _orig_attn_fwd(self, src, src_t, seq, seq_t, seq_e, mask)

    AttnModel.forward = counting
else:
    # row counts depend only on the tem_conv recursion shape, not on the attention
    # arithmetic, so stubbing the math keeps the default run fast
    def stub(self, src, src_t, seq, seq_t, seq_e, mask):
        rows_seen.append(int(src.shape[0]))
        return torch.zeros(int(src.shape[0]), int(src.shape[1])), \
               torch.zeros(int(src.shape[0]), int(mask.shape[1]))
    AttnModel.forward = stub


if args.dedup_detail:
    tem_log = []
    orig_tem = model.tem_conv

    def tem_tracking(u_emb, time, curr_layers, num_neighbors):
        tem_log.append((len(u_emb), curr_layers, int(np.unique(u_emb).shape[0])))
        return orig_tem(u_emb, time, curr_layers, num_neighbors)
    model.tem_conv = tem_tracking

print(f'\n{"=" * 88}')
print(f'{"config":<32}{"attn_calls":>11}{"total_rows":>13}{"peak_rows":>12}'
      f'{"peak_GiB":>10}{"vs_base":>9}')
print(f'{"=" * 88}')

baseline = None
for B in args.bs:
    for depth in (args.n_layer, min(args.n_layer, 1)):
        for dedup in (False, True):
            rows_seen.clear()
            model.two_hop_depth, model.row_dedup = depth, dedup
            model.eval()
            bidx = np.arange(1, B + 1)
            fake = np.random.default_rng(0).integers(1, max_idx + 1, B)
            with torch.no_grad():
                model(src_l, dst_l, fake, ts_l, e_idx_l, bidx, 1,
                      training=False, num_neighbors=K)
            total, peak = sum(rows_seen), max(rows_seen)
            gib = peak * K * MODEL_DIM * 2 / 2**30
            if depth == args.n_layer and dedup is False:
                if baseline is None:
                    baseline = peak
                ratio = f'{peak / baseline:.2f}x'
            else:
                ratio = '-'
            print(f'{f"bs={B} depth={depth} dedup={dedup}":<32}{len(rows_seen):>11}'
                  f'{total:>13,}{peak:>12,}{gib:>10.2f}{ratio:>9}')

print(f'\npeak_GiB = one [{K}, {MODEL_DIM}] fp16 projection over peak_rows rows')
print(f'{"=" * 88}')

if args.dedup_detail:
    print('\nper-call dedup compression (unique (node, cut_time) rows):')
    model.two_hop_depth, model.row_dedup = min(args.n_layer, 1), True
    tem_log.clear()
    model.eval()
    with torch.no_grad():
        model(src_l, dst_l, np.random.default_rng(0).integers(1, max_idx + 1, 20),
              ts_l, e_idx_l, np.arange(1, 21), 1, training=False, num_neighbors=K)
    tot_in = tot_uniq = 0
    for rows, depth, nuniq in tem_log:
        tot_in += rows
        tot_uniq += nuniq
        tag = f'  <-- {rows / max(nuniq, 1):.1f}x' if rows > nuniq else ''
        print(f'  curr_layers={depth}  rows={rows:>7,}  unique_nodes={nuniq:>6,}{tag}')
    print(f'  TOTAL     rows={tot_in:>8,}  unique={tot_uniq:>8,}  '
          f'overall={tot_in / max(tot_uniq, 1):.2f}x')

if args.real:
    print('\nnumerical smoke test (real attention math):')
    for depth in (args.n_layer, min(args.n_layer, 1)):
        for uniform in (False, True):
            torch.manual_seed(2020)
            np.random.seed(2020)
            m = TGIB(NeighborFinder(adj_list, uniform=uniform), n_feat, e_feat, 64,
                     num_layers=args.n_layer, use_time='time', agg_method='attn',
                     attn_mode='prod', n_head=args.n_head, drop_out=0.1)
            m.two_hop_depth, m.row_dedup = depth, True
            m.train()
            B = min(args.bs)
            losses = []
            for _ in range(args.steps):
                m.zero_grad()
                pos, neg, info = m(src_l, dst_l,
                                   np.random.default_rng(0).integers(1, max_idx + 1, B),
                                   ts_l, e_idx_l, np.arange(1, B + 1), 1,
                                   training=True, num_neighbors=K)
                loss = pos.mean() + neg.mean() + info
                loss.backward()
                gnorm = torch.sqrt(sum((v.grad ** 2).sum()
                                       for v in m.parameters() if v.grad is not None))
                ok = torch.isfinite(loss).item() and all(
                    torch.isfinite(v).all().item() for v in (pos, neg))
                losses.append((float(loss), float(gnorm), ok))
            status = 'OK' if all(o for _, _, o in losses) else 'NON-FINITE'
            print(f'  depth={depth} uniform={str(uniform):5s} {status:12s} loss ' +
                  ' '.join(f'{lo:.4f}' for lo, _, _ in losses) + '  |g| ' +
                  ' '.join(f'{gn:.2f}' for _, gn, _ in losses))
            del m
    print('\nnote: finite losses here do NOT establish an accuracy comparison;')
    print('      that needs full training runs with real validation AP/AUC.')