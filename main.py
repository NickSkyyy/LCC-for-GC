import datetime
import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from evaluate import *
from model import *
from train import *
from util import *

# seed (default 123)
np.random.seed(123)
torch.manual_seed(123)

# parameters
parser = ArgumentParser("U2GNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")

parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
# parser.add_argument("--cmp", default=0, type=int, help="if compressed")
parser.add_argument("--dataset", default="MUTAG", help="Name of the dataset")
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--edge_cmp", default=0, type=int, help="if edge-level compressions")
parser.add_argument("--edge_info", default=0, type=int, help="if adding edge-level nformation")
parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--folds", default=10, type=int, help="K-folds for train & test")
parser.add_argument("--hidden_size", default=1024, type=int, help="Hidden layer size s in [128, 256, 512, 1024]")
parser.add_argument("--K", default=1, type=int, help="Number of U2GNN layers in [1, 2, 3]")
parser.add_argument("--learning_rate", default=1e-4, type=float , help="Learning rate lr in [5e-5, 1e-4, 5e-4. 1e-3]")
parser.add_argument("--N", default=4, type=int, help="Number of neighbors in [4, 8, 16]")
parser.add_argument("--model", default="CMPX", help="Model name")
parser.add_argument("--node_cmp", default=0, type=int, help="if node-level compressions")
parser.add_argument("--name", default="PRO1")
parser.add_argument("--root", default="./")
parser.add_argument("--T", default=1, type=int, help="Number of steps T in [1, 2, 3, 4]")

parser.add_argument("--cmp_type", default="cmpx")

parser.add_argument("--std", default=0, type=float, help="Noise Ratio")
parser.add_argument("--age", default=0, type=float, help="Removal Ratio")
parser.add_argument("--test_size", default=1000, type=int, help="TEST dataset size")

args = parser.parse_args()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(123)

# logs preparation
logs_dir = os.path.abspath(os.path.join(args.root, "./logs"))
if not os.path.exists(logs_dir):
  os.makedirs(logs_dir)

logging.basicConfig(filemode='a')

formats = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler = logging.FileHandler(logs_dir + "/info.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formats)

logs = logging.getLogger()
logs.setLevel(logging.INFO)
logs.addHandler(file_handler)

logs.info(device)
logs.info(args)

# load data
degree4label = False
start_time = time.time()
origin, graphs, line_graphs, num_classes = load_data(args, args.dataset, degree4label)
end_time = time.time()
print("all pre", end_time - start_time)
if args.dataset == "TOY":
  train_set, test_set = [[0, 1, 2, 3]], [[0, 1, 2, 3]]
else:
  train_set, test_set = separate_data(origin, args.folds)
feature_size = origin[0].feature_nodes.shape[1]

# model
if args.model == "CMPX":
  model = CMPX(args, dropout=args.dropout, feature_size=feature_size, hidden_size=args.hidden_size,
              num_ATT_layers=args.T, num_classes=num_classes, num_GNN_layers=args.K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
batches_per_epoch = int((len(train_set) - 1) / args.batch_size) + 1
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=batches_per_epoch, gamma=0.1)

# main loop
out_dir = "./logs/{}".format(args.dataset)
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
logs.info("writing to {}".format(out_dir))
out = open(out_dir + "/{:%Y-%m-%d %H%M%S}.txt".format(datetime.datetime.now()), 'w')

avg_loss = []
max_valid_idx = -1
max_valid_acc = 0
min_valid_bias = 100
accs = []
for epoch in range(args.epochs):
  times = []
  losses = []
  valid_accs = []
  test_accs = []
  logs.info("# epoch %d" % epoch)
  for i in range(args.folds):
    train_fold = [[] for _ in range(3)]
    test_fold = [[] for _ in range(3)]
    valid_fold = [[] for _ in range(3)]

    if args.dataset == "TOY":
      train_fold[0] = [origin[x] for x in train_set[0]]
      valid_fold[0] = [origin[x] for x in train_set[0]]
      test_fold[0] = [origin[x] for x in train_set[0]]
    else:
      train_fold[0] = [origin[x] for x in train_set[i]]
      test_fold[0] = [origin[x] for x in test_set[i]]
      train_train_set, train_valid_set = separate_data(train_fold[0], 9)
      valid_fold[0] = [train_fold[0][x] for x in train_valid_set[0]]
      train_fold[0] = [train_fold[0][x] for x in train_train_set[0]]

    if args.node_cmp == 1:
      if args.dataset == "TOY":
        train_fold[1] = [graphs[x] for x in train_set[0]]
        valid_fold[1] = [graphs[x] for x in train_set[0]]
        test_fold[1] = [graphs[x] for x in train_set[0]]
      else:
        train_fold[1] = [graphs[x] for x in train_set[i]]
        test_fold[1] = [graphs[x] for x in test_set[i]]
        valid_fold[1] = [train_fold[1][x] for x in train_valid_set[0]]
        train_fold[1] = [train_fold[1][x] for x in train_train_set[0]]

    if args.edge_info == 1 or args.edge_cmp == 1:
      train_fold[2] = [line_graphs[x] for x in train_set[i]]
      test_fold[2] = [line_graphs[x] for x in test_set[i]]
      valid_fold[2] = [train_fold[2][x] for x in train_valid_set[0]]
      train_fold[2] = [train_fold[2][x] for x in train_train_set[0]]

    start = time.time()
    if args.model == "CMPX":
      train_loss, origin_temp, origin_graph_temp = train(model, train_fold[0], train_fold[1], train_fold[2], num_classes, args, optimizer, device)

      end = time.time()
      print("train time", end - start)
      valid_acc = evaluate(model, valid_fold[0], valid_fold[1], valid_fold[2], args, device)

      start_test = time.time()
      test_acc = evaluate(model, test_fold[0], test_fold[1], test_fold[2], args, device)
      end_test = time.time()
      print("test", end_test - start_test)
    end = time.time()

    times.append(end - start)
    losses.append(train_loss)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)

    logs.info("{epoch %d fold %d train %d valid %d test %d time %.2f loss %.2f valid_acc %.2f test_acc %.2f}" %
          (epoch, i, len(train_fold[0]), len(valid_fold[0]), len(test_fold[0]), end - start, train_loss, valid_acc * 100, test_acc * 100))

  avg_time = np.mean(times)
  avg_loss.append(np.mean(losses))
  avg_acc = round(np.mean(valid_accs) * 100, 2)
  bias = round(np.std(valid_accs) * 100, 2)
  if (avg_acc > max_valid_acc) or (avg_acc == max_valid_acc and bias <= min_valid_bias):
    max_valid_idx = epoch
    max_valid_acc = avg_acc
    min_valid_bias = bias
  accs.append(test_accs)
  logs.info("# avg. time/loss/acc: %.2f/%.2f/%.2f±%.2f" % (avg_time, avg_loss[-1], avg_acc, bias))

  if epoch > 5 and avg_loss[-1] > np.mean(avg_loss[-6:-1]):
    schedular.step()

  out.write("epoch\t" + str(epoch) + "\tvalid_acc\t" + str(avg_acc) + "\tbias\t" + str(bias) + "\n")

best_acc = round(np.mean(accs[max_valid_idx]) * 100, 2)
best_bias = round(np.std(accs[max_valid_idx]) * 100, 2)
out.write("best epoch\t" + str(max_valid_idx) + "\tbest_test_acc\t" + str(best_acc) + "\tbest_test_bias\t" + str(best_bias) + "\n")
out.close()
logs.info("done")