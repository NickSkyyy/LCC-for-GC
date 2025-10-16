# LCC-for-GC
Code for *An Efficient Loop and Clique Coarsening Algorithm for Graph Classification* (LCC4GC).

LCC4GC is an efficient loop and clique coarsening algorithm with linear complexity for graph classification on Graph Transformers architecture.
It constructs three views step by step, the original graph view, the coarsening view via **L**oop and **C**lique **C**oarsening (LCC), and the conversion view via **L**ine **G**raph **C**onversion (LGC).
Experimental results on eight real-world datasets demonstrate the improvements of LCC4GC over 31 baselines from various architectures.

## Requirement
For the dependencies required by this code, please refer to `requirements.txt`.

## Quick Start
For quick start, you could run the code:
```bash
python main.py 
  --dataset <dataset name> 
  --K <the number of model layers> 
  --T <the number of iteration steps> 
  --N <the number of sampling neighbors> 
  --hidden_size <the hidden size> 
  --node_cmp <is LCC available> 
  --edge_info <is LGC available>
```
For example,
```bash
python main.py --dataset IMDBBINARY --K 1 --T 2 --N 16 --hidden_size 1024 --node_cmp 1 --edge_info 1
```

## Detailed Settings
- dataset: supported dataset names, `COLLAB`, `IMDBBINARY`, `IMDBMULTI`, `DD`, `NCI1`, `NCI109`, `PTC`, `PROTEINS`
  - we also provide `TEST` to build the synthetic dataset
- K: the number of model layers, in `[1, 2, 3]`
- T: the number of iteration steps, in `[1, 2, 3, 4]`
- N: the number of sampling neighbors, in `[4, 8, 16]`
- hidden_size: the hidden size, in `[128, 256, 512, 1024]`
- node_cmp: if LCC is available, set `1`, otherwise `0`
- edge_info: if LGC is available, set `1`, otherwise `0`

## Optimal Parameter Settings
| dataset | COLLAB | IMDBBINARY | IMDBMULTI | DD | NCI1 | NCI109 | PTC | PROTEINS |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| K | 1 | 1 | 3 | 2 | 1 | 3 | 2 | 1 |
| T | 2 | 2 | 1 | 2 | 2 | 4 | 1 | 1 |
| N | 16 | 16 | 4 | 16 | 16 | 16 | 4 | 8 |

## Advanced Settings
In addition to the above settings, there are several parameters that need to be explained.
```bash
python main.py
  --test_size <node number of each synthetic graph>
  --std <the noise injection ratio>
  --age <the edge removal ratio>
  --cmp_type <coarsening algorithm name>
```
- test_size: node number of each synthetic dataset graph, in `[100, 1000, 10000]`
- std: the noise injection ratio, in `[0, 0.1, 0.2, 0.3, 0.4, 0.5]`
- age: the edge removal ratio, in `[0, 0.1, 0.2, 0.3, 0.4, 0.5]`
- cmp_type: supported coarsening algorithm names
  - Neighbor: `1hop`
  - Random: `rand`
  - NetworkX (NtX): `nxcyc`, `nxcli`
  - KGC: `gwnei`, `gwcli`
  - L19: `l19nei`, `l19cli`

In order to correctly implement the last two types of algorithms, please configure them according to the guidance of KGC and L19.

```bash
mkdir algorithms
cd algorithms
git clone https://github.com/ychen-stat-ml/GW-Graph-Coarsening.git gw
git clone https://github.com/loukasa/graph-coarsening.git l19
```
**Note: some of the codes in both sub-models use funtions from an earlier version, which need to be fixed slightly to align with the latest version.**

## Acknowledgements
We express our gratitude for the open-source codes provided by [U2GNN](https://github.com/daiquocnguyen/Graph-Transformer), [GarphMAE](https://github.com/THUDM/GraphMAE), [KGC](https://github.com/ychen-stat-ml/GW-Graph-Coarsening) and [L19](https://github.com/loukasa/graph-coarsening/tree/v1.1).