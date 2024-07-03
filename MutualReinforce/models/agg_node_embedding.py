import argparse
import configparser
import os
import os.path as osp
import random

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
# from torch_geometric.utils.convert import from_networkx
from torch.nn import Linear
from torch_geometric.nn import APPNP

from utils.utils import get_processed_dir


def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    attr_li = ["score", "type"]

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        # for key, value in feat_dict.items():
        for key in attr_li:
            value = feat_dict[key]
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.lin1 = Linear(args.in_channels, args.out_channels)
        # self.lin2 = Linear(args.hidden, args.out_channels)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        return x


def get_user_embeddings(filename, all_G_triple, args, model):
    G_triple = all_G_triple[filename]
    data = from_networkx(G_triple[0])
    data.x = torch.normal(mean=0, std=1e-2, size=(len(data.type), args.user_embed_dim))
    out = model(data)
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=21, help='random seed')

    parser.add_argument('--config_file', type=str, required=True)

    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--in_channels', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # TODO!!
    config = configparser.ConfigParser()
    config.read(osp.join("config", args.config_file))
    exp_name = config["TEST"].get("exp_name", None)
    args.dataset = config["KGAT"].get("dataset", "politifact")
    args.user_embed_dim = config["KGAT"].getint("user_embed_dim")

    processed_dir = get_processed_dir(exp_name)

    all_G_triple = torch.load(os.path.join(processed_dir, f"{args.dataset}_G_triple.pt"))

    model = Net(args)

    filename = "politifact14128"

    get_user_embeddings(filename, all_G_triple, args, model)


if __name__ == "__main__":
    main()
