import argparse
import os
import random
import traceback

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, ChebConv, global_mean_pool, global_max_pool  # noqa

torch.backends.cudnn.enable=True
torch.backends.cudnn.benchmark=True

from preprocess.fnn_user_dataset import FNNUserDataset


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=64, num_classes=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(train_dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()

        # 2. Readout layer
        x = global_max_pool(x, data.batch)  # [batch_size, hidden_channels]


        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin(x)
        return out

def pre_rec_f1_acc(y, y_pred):
    pre = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1  = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    return pre, rec, f1, acc

def get_train_test_split(args):
    labels_d = torch.load(os.path.join(args.root, args.dataset, 'raw',  f"{args.dataset}_labels.pt"))

    filenames_real = [x for x, y in labels_d.items() if y == 0]
    filenames_fake = [x for x, y in labels_d.items() if y == 1]

    random.shuffle(filenames_real)
    random.shuffle(filenames_fake)

    n_train_real = int(len(filenames_real) * (1-args.test_size))
    n_train_fake = int(len(filenames_fake) * (1-args.test_size))

    # Note: split should be fixed
    filenames_train = filenames_real[:n_train_real] + filenames_fake[n_train_fake:]
    filenames_test = filenames_real[n_train_real:] + filenames_fake[n_train_fake:]
    return filenames_train, filenames_test

def train(train_loader):

    model.train()

    total_loss = total_examples = 0
    preds, ys = [], []
    criterion = torch.nn.CrossEntropyLoss()
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        preds.extend(out.argmax(dim=1).tolist())
        ys.extend(data.y.tolist())

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    # print(preds)
    # print(ys)
    results = pre_rec_f1_acc(ys, preds)
    return total_loss / total_examples, results


@torch.no_grad()
def test(testloader):
    model.eval()

    ys, preds = [], []

    for data in testloader:
        data = data.to(device)
        out = model(data)
        preds.extend(out.argmax(dim=1).tolist())
        ys.extend(data.y.tolist())

    return pre_rec_f1_acc(ys, preds)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="politifact", choices=["politifact", "gossipcop"],
                        help='which dataset to use')
    parser.add_argument('--strategy', type=str, default="duplicate", choices=["copy", "drop", "random"],
                        help='Use what strategy to fill the missing values')
    parser.add_argument('--outdir', default="outputs", help='path to output directory')
    parser.add_argument('--root', default="data", help='path to store the raw dataset files')
    parser.add_argument('--debug', default=False, help='Debug')
    parser.add_argument('--make_graph', default=True, help='Construct Networkx graph?')
    parser.add_argument('--normalize_features', default=True, help='Normalize user features?')
    parser.add_argument('--theta', default=1,
                        help='Smoothing factor for calculating user impact. This ensures that user impact is always >=0')

    parser.add_argument('--epsilon', default=1e-3,
                        help='Smoothing factor to calculate relative user impact. To ensure that user impact is nonzero')
    parser.add_argument('--test_size', default=0.25,
                        help='Ratio of the dataset to be used as test set')
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--use_gdc', action='store_true')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU for debugging.')
    args = parser.parse_args()



    filenames_train, filenames_test = get_train_test_split(args)

    train_dataset = FNNUserDataset(root='.', args=args, split='train', filename_li=filenames_train, transform=T.NormalizeFeatures())
    test_dataset = FNNUserDataset(root='.', args=args, split='test', filename_li=filenames_test, transform=T.NormalizeFeatures())

    # TODO
    # pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    model = GCN().to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.

    best_val_acc = test_acc = 0
    for epoch in range(args.num_train_epochs):
        loss, (p_train, _, _, acc_train) = train(train_loader)
        p_test, _, _, acc_test = test(test_loader)
        print('Epoch: {:02d}, Loss: {:.4f} | Train: P {:.4f} Acc {:.4f}| Test: P {:.4f} Acc {:.4f}'.format(
            epoch+1, loss, p_train, acc_train, p_test, acc_test))
