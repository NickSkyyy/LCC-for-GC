from model import *
from util import *
  
def label_smoothing(labels: torch.Tensor, num_classes, smoothing=0.1):
  """
  smoothing: [0, 1). 0 for 1-hot, others for smoothing (default 0.1)
  """
  assert 0 <= smoothing < 1
  conf = 1.0 - smoothing
  shape = torch.Size((labels.size(0), num_classes))
  with torch.no_grad():
    smooth = torch.empty(size=shape, device=labels.device)
    smooth.fill_(smoothing / (num_classes - 1))
    smooth.scatter_(1, labels.data.unsqueeze(1), conf)
  return smooth

def train(model, origin, graphs, line_graphs, num_classes, args, optimizer, device):
  model.train()
  losses = 0
  indices = np.arange(0, len(origin))
  np.random.shuffle(indices)

  for l in tqdm(range(0, len(origin), args.batch_size)):
    r = l + args.batch_size
    selected = indices[l:r]
    x = [None for _ in range(3)]
    gp = [None for _ in range(3)]
    fea = [None for _ in range(3)]
    batch_graphs = [origin[i] for i in selected]
    x[0], gp[0], fea[0], labels = get_batch_data(batch_graphs, device, args.N)

    if args.node_cmp == 1:
      batch_cmp = [graphs[i] for i in selected] 
      x[1], gp[1], fea[1], _ = get_batch_data(batch_cmp, device, args.N)

    if args.edge_info == 1 or args.edge_cmp == 1:
      batch_lg = [line_graphs[i] for i in selected]
      x[2], gp[2], fea[2], _ = get_batch_data(batch_lg, device, args.N)

    labels = label_smoothing(labels, num_classes)
    optimizer.zero_grad()
    # prediction = model(x[0], gp[0], fea[0])
    # prediction = model(x[0], gp[0], fea[0], x[1], gp[1], fea[1])
    prediction, origin_temp, origin_graph_temp = model(x[0], gp[0], fea[0], x[1], gp[1], fea[1], x[2], gp[2], fea[2])
    loss = cross_entropy(prediction, labels)
    loss.backward()
    nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    losses += loss.item()
  return losses, origin_temp, origin_graph_temp