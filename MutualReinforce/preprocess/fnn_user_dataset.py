import os
import os.path as osp
from typing import Optional, Callable, List

import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from torch_geometric.data import InMemoryDataset, Data


class FNNUserDataset(InMemoryDataset):
    def __init__(self, root: str, filename_li:list,split: str = "train",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, args=None):
        self.args = args
        self.name = args.dataset
        self.debug = args.debug
        self.filename_li = filename_li
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        assert self.split in ['train', 'test']


    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, '..', '..', 'fake_news_data')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        print(f"Processing {self.args.dataset}")
        all_user_feat_d = torch.load(
            os.path.join(self.raw_dir, f"{self.name}_users{'_DEBUG' if self.debug else ''}.pt"))
        all_nx_graph_d = torch.load(
            os.path.join(self.raw_dir, f"{self.name}_nx_graphs{'_DEBUG' if self.debug else ''}.pt"))
        labels_d = torch.load(
            os.path.join(self.raw_dir, f"{self.name}_labels.pt"))

        data_list = []

        Ns = []
        for filename in all_user_feat_d.keys():
            if filename in self.filename_li \
                    and filename in all_user_feat_d\
                    and filename in all_nx_graph_d:
                # filename = 'politifact1185'
                print(filename)
                user_df = all_user_feat_d[filename]
                G_usr = all_nx_graph_d[filename]
                mapping = {name: j for j, name in enumerate(G_usr.nodes())}
                G_usr = nx.relabel_nodes(G_usr, mapping)
                Ns.append(G_usr.number_of_nodes())

                if self.args.normalize_features:
                    user_df = self.normalize_features(user_df)

                edge_index = torch.LongTensor(list(G_usr.edges)).t().contiguous()
                edge_weight = torch.FloatTensor([data['weight'] for _, _, data in list(G_usr.edges(data=True))])

                # TODO: root is connected to all tweets

                # Note: root has feature of all zeros
                x = torch.tensor(user_df.reset_index().drop(['id'], axis=1).values, dtype=torch.float32)
                x = F.pad(x, (0, 0, 1, 0), "constant", 1).contiguous()

                # Number of user feature records should be equal to
                # Number of user nodes
                # if x.shape[0] < len(G_usr):
                #     print(f"\t{filename} user feature size mismatch")
                #     m = nn.ReplicationPad2d((0, 0, 0, len(G_usr) - x.shape[0]))
                #     x = m(x)
                # elif x.shape[0] > len(G_usr):
                #     print(f"\t{filename} user feature size mismatch")
                #     x = x[:len(G_usr)].contiguous()
                # assert x.shape[0] == len(G_usr)
                if not x.shape[0] == len(G_usr):
                    print(f"\t{filename} user feature size mismatch")
                    continue

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, filename=filename, y=labels_d[filename])
                data.num_nodes = Ns[-1]

                data = data if self.pre_transform is None else self.pre_transform(data)

                data_list.append(data)

        data = self.collate(data_list)
        torch.save(data, self.processed_paths[0])

    ############################################
    # Feature Standardization
    ############################################
    def normalize_features(self, users_df):
        # TODO: move feature normalization to fnn_user_dataset
        # Feature normalization
        cols_bools = users_df.select_dtypes(include='bool').columns
        users_df.fillna({col: False for col in cols_bools}, inplace=True)
        users_df[cols_bools] = users_df[cols_bools].astype(int)
        x = users_df.values  # numpy array
        scaler = preprocessing.StandardScaler()
        x_scaled = scaler.fit_transform(x)
        users_df_scaled = pd.DataFrame(x_scaled, index=users_df.index)
        return users_df_scaled

    def __repr__(self) -> str:
        return f'{self.name}()'
