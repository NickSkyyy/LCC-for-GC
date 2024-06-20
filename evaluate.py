from model import *
from util import *
from sklearn.metrics import roc_auc_score

def evaluate(model, origin, graphs, line_graphs, args, device):
  model.eval()
  with torch.no_grad():
    predictions = []
    idx = np.arange(len(origin))
    for i in range(0, len(origin), args.batch_size):
      selected = idx[i:i + args.batch_size]
      if len(selected) == 0:
        continue
      x = [None for _ in range(3)]
      gp = [None for _ in range(3)]
      fea = [None for _ in range(3)]
      batch_graphs = [origin[j] for j in selected]
      x[0], gp[0], fea[0], labels = get_batch_data(batch_graphs, device, args.N)

      if args.node_cmp == 1:
        batch_cmp = [graphs[j] for j in selected]
        x[1], gp[1], fea[1], _ = get_batch_data(batch_cmp, device, args.N)

      if args.edge_info == 1 or args.edge_cmp == 1:
        batch_lg = [line_graphs[j] for j in selected]
        x[2], gp[2], fea[2], _ = get_batch_data(batch_lg, device, args.N)
      
      # prediction = model(x[0], gp[0], fea[0]).detach()
      # prediction = model(x[0], gp[0], fea[0], x[1], gp[1], fea[1]).detach()
      prediction, origin_temp, origin_graph_temp = model(x[0], gp[0], fea[0], x[1], gp[1], fea[1], x[2], gp[2], fea[2])
      prediction = prediction.detach()
      predictions.append(prediction)
  predictions = torch.cat(predictions, 0)
  predictions = predictions.max(1, keepdim=True)[1]
  labels = torch.LongTensor([graph.label for graph in origin]).to(device)
  acc = (predictions.eq(labels.view_as(predictions)).sum().cpu().item()) / float(len(origin))
  # acc = roc_auc_score(labels.view_as(predictions).cpu().numpy(), predictions.cpu().numpy())
  return acc