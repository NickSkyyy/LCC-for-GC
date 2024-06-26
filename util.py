import copy
import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 300
import networkx as nx
import numpy as np
import os
import random
import re
import time
import torch
import torch.nn as nn
import sys
sys.setrecursionlimit(4000)

from collections import defaultdict
from algorithms.gw import coarsening_utils as cu
from algorithms.l19.libraries import coarsening_utils as lcu
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset
from tqdm import tqdm

class MyGraph(object):
  def __init__(self, graph, label, label_nodes=None, feature_nodes=None):
    """
    graph: graph of networkx format
    label: graph label
    label_nodes: [label for node 0, label for node 1, ...]
    feature_nodes: [feature vector for node 0, feature vector for node 1, ...] 
    sparse_A: [[neighbor 0 for node 0], [1 for node 0], ...], [x for node 1, ...], ...]
    edge_mat: TODO
    """
    self.graph = graph
    self.label = label
    self.label_nodes = label_nodes
    self.feature_nodes = feature_nodes 
    self.sparse_A = None
    self.edge_mat = None

def cross_entropy(prediction, soft):
  logSoftMax = nn.LogSoftmax(dim=1)
  return torch.mean(torch.sum(- soft * logSoftMax(prediction), 1))

def get_batch_data(graphs, device, N):
  """
  get batch data from batch graphs
  ### return  
  - inputs: A x B, A the number of neighbors + 1, B is the number of all nodes in graphs
  - graph_pool: A x B, A is the number of graphs in batch, B is the number of all nodes in graphs (A)
  - concat: A x B, A is the number of all nodes in graphs, B is the dimension of features of nodes
  """
  concat = np.concatenate([graph.feature_nodes for graph in graphs])
  concat = torch.from_numpy(concat).to(device)
  # concat_cmp = np.concatenate([graph.feature_nodes for graph in graphs_cmp])
  # concat_cmp = torch.from_numpy(concat_cmp).to(device)
  # for i in range(len(graphs[0].graph)):
  #   print(concat[i])

  # get graph pool & edge mat
  sum = [0]
  # edge_mat = graph.edge_mat (+ sum[i])
  edge_mat = []
  for i, graph in enumerate(graphs):
    sum.append(sum[i] + len(graph.graph))
    edge_mat.append(graph.edge_mat + sum[i])

  # sum_cmp = [0]
  # edge_mat_cmp = []
  # for i, graph in enumerate(graphs_cmp):
  #   sum.append(sum_cmp[i] + len(graph.graph))
  #   edge_mat_cmp.append(graph.edge_mat + sum_cmp[i])

  # graph pool
  # indices is the list of [graph_id, node_id]
  indices = []
  # item is [1] with the number of all nodes
  item = []
  for i, graph in enumerate(graphs):
    item.extend([1] * len(graph.graph))
    indices.extend([[i, j] for j in range(sum[i], sum[i + 1])])
  item = torch.FloatTensor(item)
  indices = torch.LongTensor(indices).transpose(0, 1)
  graph_pool = torch.sparse_coo_tensor(indices, item, torch.Size([len(graphs), sum[-1]])).to(device)

  # edge mat for adjacent matrix
  block = np.concatenate(edge_mat, 1)
  rows = block[0,:]
  cols = block[1,:]
  sparse_A = {}
  for i in range(len(rows)):
    if rows[i] not in sparse_A:
      sparse_A[rows[i]] = []
    sparse_A[rows[i]].append(cols[i])
  
  neighbors = []
  for node in range(concat.shape[0]):
    if node in sparse_A:
      neighbors.append([node] + list(np.random.choice(sparse_A[node], N, replace=True)))
    else:
      neighbors.append([node for _ in range(N + 1)])
  
  inputs = np.array(neighbors)
  inputs = torch.transpose(torch.from_numpy(inputs), 0, 1).to(device)
  # for i in range(len(graphs[0].graph)):
  #   print(inputs[:, i])
  
  labels = np.array([graph.label for graph in graphs])
  labels = torch.from_numpy(labels).to(device).type(torch.int64)

  # print(inputs.shape)
  # print(graph_pool.shape)
  # print(concat.shape)

  return inputs, graph_pool, concat, labels

def load_data(args, dataset, degree4label=False):
  """
  ### Parameters
  - args: all settings
  - dataset: name of dataset
  - degree4label: true if degree for node label
  ### Return
  - origin: graphs
  - graphs: node-level compressions
  - line_graphs: edge-level compressions
  - label_graph2num: number of classes
  """
  print("loading data %s" % dataset)

  graphs = []
  label_graph2num = {}
  label_node2num = {}
  nodes = 0
  edges = 0
  degrees = 0
  
  if dataset == "MalNet":
    categories = ["add", "adw", "ben", "dow", "tro"]
    print("load MalNet train")
    with open("./dataset/MalNet/split_info_tiny/type/train.txt", "r") as f:
      while True:
        fname = f.readline().strip()
        if fname == "":
          break
        label_graph = categories.index(fname[:3])
        if label_graph not in label_graph2num:
          val = len(label_graph2num)
          label_graph2num.setdefault(label_graph, val)
        with open("./dataset/MalNet/malnet-graphs-tiny/%s.edgelist" % fname, 'r') as g:
          for _ in range(3):
            g.readline()
          nnode, nedge = re.findall("\d+", g.readline())
          nodes += int(nnode)
          edges += int(nedge)
          g.readline()

          # build graph
          graph = nx.Graph()
          label_nodes = []
          feature_nodes = []
          edge_list = []
          ss = set()
          for _ in range(int(nedge)):
            u, v = g.readline().strip().split()
            ss.add(int(u))
            ss.add(int(v))
            edge_list.append([int(u), int(v)])
            # edge_list.append([int(v), int(u)])
          ss = sorted(list(ss))
          for node in range(len(ss)):
            graph.add_node(node, label="%d" % node)
          graph.add_edges_from([[ss.index(u), ss.index(v)] for u, v in edge_list])
          feature_nodes = None
          label_node = 0
          if label_node not in label_node2num:
            val = len(label_node2num)
            label_node2num.setdefault(label_node, val)
          label_nodes.append(label_node2num.get(label_node))
          degrees += np.mean(list(dict(graph.degree).values()))
          graphs.append(MyGraph(graph, label_graph2num.get(label_graph), label_nodes, feature_nodes))
    print("load MalNet valid")
    with open("./dataset/MalNet/split_info_tiny/type/val.txt", "r") as f:
      while True:
        fname = f.readline().strip()
        if fname == "":
          break
        label_graph = categories.index(fname[:3])
        if label_graph not in label_graph2num:
          val = len(label_graph2num)
          label_graph2num.setdefault(label_graph, val)
        with open("./dataset/MalNet/malnet-graphs-tiny/%s.edgelist" % fname, 'r') as g:
          for _ in range(3):
            g.readline()
          nnode, nedge = re.findall("\d+", g.readline())
          nodes += int(nnode)
          edges += int(nedge)
          g.readline()

          # build graph
          graph = nx.Graph()
          label_nodes = []
          feature_nodes = []
          edge_list = []
          ss = set()
          for _ in range(int(nedge)):
            u, v = g.readline().strip().split()
            ss.add(int(u))
            ss.add(int(v))
            edge_list.append([int(u), int(v)])
            # edge_list.append([int(v), int(u)])
          ss = sorted(list(ss))
          for node in range(len(ss)):
            graph.add_node(node, label="%d" % node)
          graph.add_edges_from([[ss.index(u), ss.index(v)] for u, v in edge_list])
          feature_nodes = None
          label_node = 0
          if label_node not in label_node2num:
            val = len(label_node2num)
            label_node2num.setdefault(label_node, val)
          label_nodes.append(label_node2num.get(label_node))
          degrees += np.mean(list(dict(graph.degree).values()))
          graphs.append(MyGraph(graph, label_graph2num.get(label_graph), label_nodes, feature_nodes))
    print("load MalNet test")
    with open("./dataset/MalNet/split_info_tiny/type/test.txt", "r") as f:
      while True:
        fname = f.readline().strip()
        if fname == "":
          break
        label_graph = categories.index(fname[:3])
        if label_graph not in label_graph2num:
          val = len(label_graph2num)
          label_graph2num.setdefault(label_graph, val)
        with open("./dataset/MalNet/malnet-graphs-tiny/%s.edgelist" % fname, 'r') as g:
          for _ in range(3):
            g.readline()
          nnode, nedge = re.findall("\d+", g.readline())
          nodes += int(nnode)
          edges += int(nedge)
          g.readline()

          # build graph
          graph = nx.Graph()
          label_nodes = []
          feature_nodes = []
          edge_list = []
          ss = set()
          for _ in range(int(nedge)):
            u, v = g.readline().strip().split()
            ss.add(int(u))
            ss.add(int(v))
            edge_list.append([int(u), int(v)])
            # edge_list.append([int(v), int(u)])
          ss = sorted(list(ss))
          for node in range(len(ss)):
            graph.add_node(node, label="%d" % node)
          graph.add_edges_from([[ss.index(u), ss.index(v)] for u, v in edge_list])
          feature_nodes = None
          label_node = 0
          if label_node not in label_node2num:
            val = len(label_node2num)
            label_node2num.setdefault(label_node, val)
          label_nodes.append(label_node2num.get(label_node))
          degrees += np.mean(list(dict(graph.degree).values()))
          graphs.append(MyGraph(graph, label_graph2num.get(label_graph), label_nodes, feature_nodes))
  elif dataset == "Peptides-func":
    train = LRGBDataset("./dataset/Peptides-func", "Peptides-func", "train")
    valid = LRGBDataset("./dataset/Peptides-func", "Peptides-func", "val")
    test = LRGBDataset("./dataset/Peptides-func", "Peptides-func", "test")
    for graph in train:
      print(graph.y)
      input()
  else:
    with open("./dataset/%s/%s.txt" % (dataset, dataset), 'r') as f:
      num_graph = int(f.readline().strip())
      # for _ in range(num_graph):
      for _ in tqdm(range(num_graph)):
        row = f.readline().strip().split()
        num_node, label_graph = [int(x) for x in row]
        nodes += num_node
        # graph label
        if label_graph not in label_graph2num:
          val = len(label_graph2num)
          label_graph2num.setdefault(label_graph, val)

        graph = nx.Graph()
        label_nodes = []
        feature_nodes = []
        for node in range(num_node):
          graph.add_node(node, label="%d" % node)
          row = f.readline().strip().split()
          label_node, num_edge = [int(row[0]), int(row[1])]
          edges += num_edge
          # find attributes
          attr = None
          # if num_edge + 2 != len(row):
          #   print("find attributes")
          #   # attr = np.array([float(x) for x in row[num_edge + 2:]])
          #   attr = [float(x) for x in row[num_edge + 2:]]
          #   feature_nodes.append(attr)
          # node label
          if label_node not in label_node2num:
            val = len(label_node2num)
            label_node2num.setdefault(label_node, val)
          label_nodes.append(label_node2num.get(label_node))
          # add edges
          for k in range(2, 2 + num_edge):
            graph.add_edge(node, int(row[k]))
          # turn feature list to np format
          if attr is None:
            feature_nodes = None
          else:
            feature_nodes = np.stack(feature_nodes)

        degrees += np.mean(list(dict(graph.degree).values()))
        graphs.append(MyGraph(graph, label_graph2num.get(label_graph), label_nodes, feature_nodes))

  nodes /= len(graphs)
  edges = edges / 2 / len(graphs)
  degrees /= len(graphs)

  # build sparse_A and edge_mat
  for graph in graphs:
    graph.sparse_A = [[] for _ in range(len(graph.graph))]
    for u, v in graph.graph.edges():
      graph.sparse_A[u].append(v)
      graph.sparse_A[v].append(u)

    temp = [list(pair) for pair in graph.graph.edges()]
    temp.extend([[i, j] for j, i in temp])
    if len(temp) != 0:
      graph.edge_mat = np.transpose(np.array(temp, dtype=np.int32), (1, 0))
    else:
      graph.edge_mat = np.array([[], []], dtype=np.int32)  
    
  # make heterogeneous graph
  if len(label_node2num) == 1 or degree4label:
    print("# degree as label: True")
    temp = set()
    for graph in graphs:
      # graph.label_nodes = list(dict(graph.graph.degree).values())
      graph.label_nodes = [[k, v] for k, v in dict(graph.graph.degree).items()]
      graph.label_nodes = [kv[1] for kv in sorted(graph.label_nodes, key=lambda x : x[0])]
      temp = temp.union(set(dict(graph.graph.degree).values()))
    label_node2num = {x : i for i, x in enumerate(temp)}
    for graph in graphs:
      graph.label_nodes = [label_node2num.get(x) for x in graph.label_nodes]

  # build features (default 1-hot for label nodes)
  for graph in graphs:
    graph.feature_nodes = np.zeros((len(graph.label_nodes), len(label_node2num)), dtype=np.float32)
    for i, j in enumerate(graph.label_nodes):
      graph.feature_nodes[i][j] = 1

  # 从这里往后默认graphs是点图的压缩结果
  # # mol-HIV
  # ids = []
  # for i, graph in enumerate(graphs):
  #   if graph.label == 1:
  #     ids.append(i)
  # cnt = len(graphs) - len(ids)
  # for _ in range(cnt):
  #   ii = random.choice(ids)
  #   gg = copy.deepcopy(graphs[ii])
  #   graphs.append(gg)
  # print("now:", len(graphs))

  if args.node_cmp == 1:
    origin = copy.deepcopy(graphs)
  if args.edge_info == 1 or args.edge_cmp == 1:
    origin = copy.deepcopy(graphs)
    line_graphs = copy.deepcopy(graphs)

  def find_loop(graph, node):
    def dfs(graph, cur):
      loops = set()
      qq.append(cur)
      nodes.add(cur)
      for neighbor in graph.sparse_A[cur]:
        edge = tuple([cur, neighbor]) if cur < neighbor else tuple([neighbor, cur])
        if edge in edges:
          continue
        edges.add(edge)
        if neighbor in qq[-7:-1]:
          loops.add(tuple(sorted(qq[qq.index(neighbor):])))
          continue
        if neighbor in nodes:
          continue
        loops = loops.union(dfs(graph, neighbor))
      qq.remove(cur)
      return loops

    qq = []

    nodes = set()
    edges = set()

    return dfs(graph, node)

  def parse_group(graph, nodes: list, cur: list, nodes_vis: set):
    """
    nodes: nodes list for parsing
    cur: dfs for root, write results once the [] get
    """
    groups = []
    nodes.sort(key=lambda x : (graph.graph.degree[x], -x))
    node_cur = nodes[-1] 
    cur.append(node_cur)

    # 自环应当从邻居节点中删除
    neighbors = list(set(graph.sparse_A[node_cur]).intersection(set(nodes)) - {node_cur})
    rests = list(set(nodes) - set(neighbors) - {node_cur})

    if len(neighbors) != 0:
      group_temp, vis_temp = parse_group(graph, neighbors, cur, nodes_vis)
      groups.extend(group_temp)
      nodes_vis = nodes_vis.union(vis_temp)
    else:
      nodes_vis = nodes_vis.union(set(cur[1:]))
      if len(cur) > 2:
        cur.sort(key=lambda x : (-graph.graph.degree[x], x))
        groups.append(cur)
    if node_cur not in nodes_vis:
      rests.extend(set(neighbors) - nodes_vis)
      if len(rests) != 0:
        group_temp, vis_temp = parse_group(graph, rests, [], nodes_vis)
        groups.extend(group_temp)
        nodes_vis = nodes_vis.union(vis_temp)
      nodes_vis.add(node_cur)

    return groups, nodes_vis
  
  cases = []
  color_map = []
  for _ in range(len(label_node2num)):
    color_map.append("#" + ''.join(random.choice("0123456789ABCDEF") for i in range(6)))

  # for x in cases:
  #   color_seq = [color_map[origin[x].label_nodes[node]] for node in origin[x].graph.nodes()]
  #   nx.draw(origin[x].graph, pos=nx.spring_layout(origin[x].graph), node_color=color_seq, node_size=50)
  #   plt.show()

  NODE_CMP_TYPE = args.cmp_type
  if args.node_cmp == 1:
    start_time = time.time()
    print("node-level compressions")
    # 对点图压缩
    for idx in tqdm(range(len(graphs))):
      # save labels
      temp = {}
      for node, label in zip(graphs[idx].graph.nodes(), graphs[idx].label_nodes):
        temp.setdefault(node, {"label": label})
      nx.set_node_attributes(graphs[idx].graph, temp)

      # find nodes with most degrees
      ds = list(dict(graphs[idx].graph.degree).items())
      d2map = []
      for _, item in enumerate(ds):
        d2map.append(item[0])
      d2map.sort(reverse=True)
      groups = []

      if NODE_CMP_TYPE == "cmpx":
        try:
          groups, vis = parse_group(graphs[idx], d2map, [], set())
        except RecursionError:
          print("Error: graph %d" % idx)
      
        # for loops (begin)
        try:
          if len(groups) == 0:
            groups = find_loop(graphs[idx], d2map[0])
        except RecursionError:
          print("Error: graph %d" % idx)
      elif NODE_CMP_TYPE == "1hop":
        res = list(graphs[idx].graph.nodes())
        while len(res) != 0:
          cur = random.choice(res)
          cur = set(graphs[idx].sparse_A[cur]).union({cur})
          groups.append(list(cur))
          res = list(set(res) - cur)
      elif NODE_CMP_TYPE == "rand":
        ss = len(graphs[idx].graph.nodes()) - 1
        r = 5
        for _ in range(r):
          k = int(random.random() * ss) + 1
          pnodes = set(random.sample(list(graphs[idx].graph.nodes()), k=k))
          groups.append(list(pnodes))
      elif NODE_CMP_TYPE == "gwnei":
        adj = nx.adjacency_matrix(graphs[idx].graph).toarray()
        G = cu.gsp.graphs.Graph(adj)
        C, _, _, _ = cu.coarsen(G, K=5, max_levels=1, method="variation_neighborhood")
        dmap = defaultdict(list)
        for i in range(len(C.indices)):
          dmap[C.indices[i]].append(C.indptr[i])
        groups = list(dmap.values())
      elif NODE_CMP_TYPE == "gwcli":
        adj = nx.adjacency_matrix(graphs[idx].graph).toarray()
        G = cu.gsp.graphs.Graph(adj)
        C, _, _, _ = cu.coarsen(G, K=5, max_levels=1, method="variation_cliques")
        dmap = defaultdict(list)
        for i in range(len(C.indices)):
          dmap[C.indices[i]].append(C.indptr[i])
        groups = list(dmap.values())
      elif NODE_CMP_TYPE == "nxcli":
        try:
          cliques = nx.find_cliques(graphs[idx].graph)
        except:
          cliques = nx.find_cycle(graphs[idx].graph)
        groups = list(cliques)
      elif NODE_CMP_TYPE == "nxcyc":
        try:
          cycles = nx.find_cycle(graphs[idx].graph)
        except:
          cycles = nx.find_cliques(graphs[idx].graph)
        groups = list(cycles)
      elif NODE_CMP_TYPE == "nxde":
        deg, den = nx.dedensify(graphs[idx].graph, 2)
        if len(den) == 0:
          continue
        for item in den:
          print(item)  
        input()
      elif NODE_CMP_TYPE == "l19nei":
        adj = nx.adjacency_matrix(graphs[idx].graph).toarray()
        G = lcu.gsp.graphs.Graph(adj)
        C, _, _, _ = lcu.coarsen(G, K=5, max_levels=1, method="variation_neighborhood")
        dmap = defaultdict(list)
        for i in range(len(C.indices)):
          dmap[C.indices[i]].append(C.indptr[i])
        groups = list(dmap.values())
      elif NODE_CMP_TYPE == "l19cli":
        adj = nx.adjacency_matrix(graphs[idx].graph).toarray()
        G = lcu.gsp.graphs.Graph(adj)
        C, _, _, _ = lcu.coarsen(G, K=5, max_levels=1, method="variation_cliques")
        dmap = defaultdict(list)
        for i in range(len(C.indices)):
          dmap[C.indices[i]].append(C.indptr[i])
        groups = list(dmap.values())
      else:
        groups = []

      nodes_new = []
      fea_new = []
      vis_temp = set()
      for loop in groups:
        nodes_new.append(list(loop))
        fea_temp = graphs[idx].feature_nodes[loop[0]]
        for node in loop[1:]:
          fea_temp += graphs[idx].feature_nodes[node]
        fea_new.append(fea_temp)
        vis_temp = vis_temp.union(loop)
      temp = set(graphs[idx].graph.nodes()) - vis_temp
      for node in temp:
        nodes_new.append([node])
        fea_new.append(graphs[idx].feature_nodes[node])
      # edges
      edges_new = []    
      edges_temp = []
      edges_map = []
      for i in range(len(nodes_new)):
        edges_map.append([0 for _ in range(len(nodes_new))])
      for u in range(len(nodes_new)):
        node = nodes_new[u]
        temp = set(node)
        for item in node:
          temp = temp.union(graphs[idx].sparse_A[item])
        node_nei = temp
        for v in range(u + 1, len(nodes_new)):
          next = nodes_new[v]
          temp = set(next)
          for item in next:
            temp = temp.union(graphs[idx].sparse_A[item])
          next_nei = temp
          if len(set(node).intersection(next_nei)) != 0 or len(set(next).intersection(node_nei)) != 0:
            edges_new.append([u, v])
          # if len(set(node).intersection(next)) != 0:
          #   edges_new.append([u, v])
          #   edges_map[u][v] = 1
          #   edges_map[v][u] = 1
          # elif len(set(node).intersection(next_nei)) != 0 or len(set(next).intersection(node_nei)) != 0:
          #   edges_temp.append([u, v])
      # 检查环路是否是通过边连接，而不是通过重合点
      # 如果是边的话，需要检测是否通过其他节点进行连接，如果是，则两个新点之间不需要连边
      # 否则，点之间需要连接
      # for u, v in edges_temp:
      #   node, next = set(), set()
      #   for i, x in enumerate(edges_map[u]):
      #     if x == 1:
      #       node.add(i)
      #   for i, x in enumerate(edges_map[v]):
      #     if x == 1:
      #       next.add(i)
      #   if len(set(node).intersection(next)) == 0:
      #     edges_new.append([u, v])
      # new graph
      graph_temp = nx.Graph()
      for node, _ in enumerate(nodes_new):
        graph_temp.add_node(node, label="%d" % node)
      for u, v in edges_new:
        graph_temp.add_edge(u, v)
      # node labels of graph
      graph_temp_label_node = []
      for node in nodes_new:
        graph_temp_label_node.append(graphs[idx].graph.nodes[node[0]]["label"])
      graphs[idx].graph = graph_temp
      graphs[idx].label_nodes = graph_temp_label_node
      graphs[idx].feature_nodes = np.stack(fea_new)
      # sparse_A
      sparse_A_temp = [[] for _ in range(len(graphs[idx].graph))]
      for u, v in graphs[idx].graph.edges():
        sparse_A_temp[u].append(v)
        sparse_A_temp[v].append(u)
      graphs[idx].sparse_A = sparse_A_temp
      # edge_mat
      temp = [list(pair) for pair in graphs[idx].graph.edges()]
      temp.extend([[i, j] for j, i in temp])
      if len(temp) != 0:
        graphs[idx].edge_mat = np.transpose(np.array(temp, dtype=np.int32), (1, 0))
      else:
        graphs[idx].edge_mat = np.array([[], []], dtype=np.int32)
      # print(nodes_new)
    for x in cases:
      color_seq = [color_map[graphs[x].label_nodes[node]] for node in graphs[x].graph.nodes()]
      nx.draw(graphs[x].graph, pos=nx.spring_layout(graphs[x].graph), node_color=color_seq, node_size=50)
      plt.show()
    print("node-level compressed")
    end_time = time.time()
  lcc_time = end_time - start_time
  lcc_n, lcc_e = 0, 0
  for idx in range(len(graphs)):
    gg = graphs[idx]
    lcc_n += len(gg.graph.nodes())
    lcc_e += len(gg.graph.edges())
  print("lcc", lcc_n / len(graphs), lcc_e / len(graphs), lcc_time)
  exit(0)
  # exit(0)
  # build line graph
  if args.edge_info == 1 or args.edge_cmp == 1:
    print("build line graphs")
    start_time = time.time()
    line_graphs = copy.deepcopy(graphs)
    for idx in tqdm(range(len(line_graphs))):   
      graph = line_graphs[idx]
      tgraph = nx.line_graph(graph.graph)
      if len(tgraph) == 0:
        continue
      graph.graph = tgraph
      temp = graph.graph.nodes()
      node2label = {old : new for new, old in enumerate(graph.graph.nodes())}
      graph.graph = nx.relabel_nodes(graph.graph, node2label)
      # 先做成同质图
      graph.label_nodes = [0 for _ in range(len(graph.graph.nodes()))]
      # line graph feature nodes
      graph.feature_nodes = np.zeros(((len(graph.label_nodes), len(label_node2num))), dtype=np.float32)
      for x, (i, j) in enumerate(temp):
        # WARN: 之前采用的是origin
        # graph.feature_nodes[x] += origin[idx].feature_nodes[i] + origin[idx].feature_nodes[j]    
        graph.feature_nodes[x] += graphs[idx].feature_nodes[i] + graphs[idx].feature_nodes[j] 
      # line graph sparse_A
      graph.sparse_A = [[] for _ in range(len(graph.graph))]
      for u, v in graph.graph.edges():
        graph.sparse_A[u].append(v)
        graph.sparse_A[v].append(u)
      # line graph edge_mat
      temp = [list(pair) for pair in graph.graph.edges()]
      temp.extend([[i, j] for j, i in temp])
      if len(temp) != 0:
        graph.edge_mat = np.transpose(np.array(temp, dtype=np.int32), (1, 0))
      else:
        graph.edge_mat = np.array([[], []], dtype=np.int32) 
      line_graphs[idx] = graph
    for x in cases:
      nx.draw(line_graphs[x].graph, pos=nx.spring_layout(line_graphs[x].graph), node_size=50)
      plt.show()
    end_time = time.time()
    print("built")
  lgc_time = end_time - start_time

  lcc_n, lcc_e = 0, 0
  lgc_n, lgc_e = 0, 0
  for idx in range(len(graphs)):
    gg = graphs[idx]
    lcc_n += len(gg.graph.nodes())
    lcc_e += len(gg.graph.edges())
    gg = line_graphs[idx]
    lgc_n += len(gg.graph.nodes())
    lgc_e += len(gg.graph.edges())
  
  print("lgc", lgc_n / len(graphs), lgc_e / len(graphs), lgc_time)
  exit(0)

  if args.edge_cmp == 1:
    print("edge-level compressions")
    # 对线图压缩
    for idx in tqdm(range(len(line_graphs))):
      # save labels
      temp = {}
      for node, label in zip(line_graphs[idx].graph.nodes(), line_graphs[idx].label_nodes):
        temp.setdefault(node, {"label": label})
      nx.set_node_attributes(line_graphs[idx].graph, temp)

      # find nodes with most degrees
      ds = list(dict(line_graphs[idx].graph.degree).items())
      d2map = []
      for _, item in enumerate(ds):
        d2map.append(item[0])
      d2map.sort(reverse=True)

      groups, vis = parse_group(line_graphs[idx], d2map, [], set())
    
      # for loops (begin)
      # if len(groups) == 0:
      #   groups = find_loop(line_graphs[idx], d2map)
      # print(idx)
      # print(loops)
      # continue

      nodes_new = []
      fea_new = []
      vis_temp = set()
      for loop in groups:
        nodes_new.append(list(loop))
        fea_temp = line_graphs[idx].feature_nodes[loop[0]]
        for node in loop[1:]:
          fea_temp += line_graphs[idx].feature_nodes[node]
        fea_new.append(fea_temp)
        vis_temp = vis_temp.union(loop)
      temp = set(line_graphs[idx].graph.nodes()) - vis_temp
      for node in temp:
        nodes_new.append([node])
        fea_new.append(line_graphs[idx].feature_nodes[node])
      # edges
      edges_new = []    
      edges_temp = []
      edges_map = []
      for i in range(len(nodes_new)):
        edges_map.append([0 for _ in range(len(nodes_new))])
      for u in range(len(nodes_new)):
        node = nodes_new[u]
        temp = set(node)
        for item in node:
          temp = temp.union(line_graphs[idx].sparse_A[item])
        node_nei = temp
        for v in range(u + 1, len(nodes_new)):
          next = nodes_new[v]
          temp = set(next)
          for item in next:
            temp = temp.union(line_graphs[idx].sparse_A[item])
          next_nei = temp
          if len(set(node).intersection(next_nei)) != 0 or len(set(next).intersection(node_nei)) != 0:
            edges_new.append([u, v])
          # if len(set(node).intersection(next)) != 0:
          #   edges_new.append([u, v])
          #   edges_map[u][v] = 1
          #   edges_map[v][u] = 1
          # elif len(set(node).intersection(next_nei)) != 0 or len(set(next).intersection(node_nei)) != 0:
          #   edges_temp.append([u, v])
      # 检查环路是否是通过边连接，而不是通过重合点
      # 如果是边的话，需要检测是否通过其他节点进行连接，如果是，则两个新点之间不需要连边
      # 否则，点之间需要连接
      # for u, v in edges_temp:
      #   node, next = set(), set()
      #   for i, x in enumerate(edges_map[u]):
      #     if x == 1:
      #       node.add(i)
      #   for i, x in enumerate(edges_map[v]):
      #     if x == 1:
      #       next.add(i)
      #   if len(set(node).intersection(next)) == 0:
      #     edges_new.append([u, v])
      # new graph
      graph_temp = nx.Graph()
      for node, _ in enumerate(nodes_new):
        graph_temp.add_node(node, label="%d" % node)
      for u, v in edges_new:
        graph_temp.add_edge(u, v)
      # node labels of graph
      graph_temp_label_node = []
      for node in nodes_new:
        graph_temp_label_node.append(line_graphs[idx].graph.nodes[node[0]]["label"])
      line_graphs[idx].graph = graph_temp
      line_graphs[idx].label_nodes = graph_temp_label_node
      line_graphs[idx].feature_nodes = np.stack(fea_new)
      # sparse_A
      sparse_A_temp = [[] for _ in range(len(line_graphs[idx].graph))]
      for u, v in line_graphs[idx].graph.edges():
        sparse_A_temp[u].append(v)
        sparse_A_temp[v].append(u)
      line_graphs[idx].sparse_A = sparse_A_temp
      # edge_mat
      temp = [list(pair) for pair in line_graphs[idx].graph.edges()]
      temp.extend([[i, j] for j, i in temp])
      if len(temp) != 0:
        line_graphs[idx].edge_mat = np.transpose(np.array(temp, dtype=np.int32), (1, 0))
      else:
        line_graphs[idx].edge_mat = np.array([[], []], dtype=np.int32)
      # print(nodes_new)
    for x in cases:
      color_seq = [color_map[line_graphs[x].label_nodes[node]] for node in origin[x].graph.nodes()]
      nx.draw(line_graphs[x].graph, pos=nx.spring_layout(line_graphs[x].graph), node_color=color_seq, node_size=50)
      plt.show()
    print("edge-level compressed")

  # drawing
  # if not os.path.exists("./visual/%s" % dataset):
  #   print("saving figures")
  #   os.makedirs("./visual/%s" % dataset)
  #   # build color map
  #   color_map = []
  #   for _ in range(len(label_node2num)):
  #     color_map.append("#" + ''.join(random.choice("0123456789ABCDEF") for i in range(6)))
  #   print("# color map: %d" % len(color_map))
  #   for i in tqdm(range(len(graphs))):
  #     graph = graphs[i]
  #     color_seq = [color_map[graph.label_nodes[node]] for node in graph.graph.nodes()]
  #     nx.draw(graph.graph, pos=nx.spring_layout(graph.graph), with_labels=True, node_color=color_seq, node_size=100)
  #     # nx.draw(graph.graph, pos=nx.spring_layout(graph.graph), node_color=color_seq, node_size=100)
  #     plt.savefig("./visual/%s/%d_%d.png" % (dataset, i, graph.label))
  #     plt.close()
  # else:
  #   print("saving already")

  print("# graph classes: %d" % len(label_graph2num))
  print("# node classes: %d" % len(label_node2num))
  print("# graphs: %d" % len(graphs))
  print("# avg. nodes/edges/degrees: %.2f/%.2f/%.2f" % (nodes, edges, degrees))
  print("# features dimension: %d" % (-1 if graphs[0].feature_nodes is None else len(graphs[0].feature_nodes[0])))
  
  print("%s loaded\n" % dataset)
  output = [graphs, None, None]
  if args.node_cmp == 1:
    output[0] = origin
    output[1] = graphs
  if args.edge_info == 1 or args.edge_cmp == 1:
    output[0] = origin
    output[2] = line_graphs
  return output[0], output[1], output[2], len(label_graph2num)

def separate_data(graphs, folds=10):
  skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
  labels = [graph.label for graph in graphs]
  train_set = []
  test_set = []
  for _, (train, test) in enumerate(skf.split(graphs, labels)):
    train_set.append(train)
    test_set.append(test)
  return train_set, test_set