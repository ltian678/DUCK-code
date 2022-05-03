import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv,GraphConv,GINConv,GATConv
import copy
from torch_scatter import scatter_mean, scatter_max, scatter_add




class SimpleGCN(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,pooling='scatter_mean',dropout=0.5):
        super(SimpleGCN, self).__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.pooling == 'scatter_mean':
          x = scatter_mean(x,data.batch,dim=0)
        elif self.pooling == 'scatter_max':
          x = scatter_max(x,data.batch,dim=0)
        elif self.pooling == 'scatter_add':
          x = scatter_add(x,data.batch,dim=0)
        elif self.pooling == 'global_mean':
          x = global_mean_pool(x,data.batch)
        elif self.pooling == 'global_max':
          x = global_max_pool(x,data.batch)
        elif self.pooling == 'mean_max':
          x_mean = global_mean_pool(x,data.batch)
          x_max = global_max_pool(x,data.batch)
          x = torch.cat((x_mean,x_max), 1)
        elif self.pooling == 'scatter_mean_max':
          x_mean = scatter_mean(x,data.batch,dim=0)
          x_max = scatter_add(x,data.batch,dim=0)
          x = torch.cat([x_mean,x_max],1)
        elif self.pooling == 'root':
          rootindex = data.rootindex
          root_extend = torch.zeros(len(data.batch), 768).to(device)
          batch_size = max(data.batch) + 1
          for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x[rootindex[num_batch]]
          x = root_extend
        else:
          assert False, "Something wrong with the parameter --pooling"

        return x


class SimpleGCNNet(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,D_H, D_out,pooling='scatter_mean'):
        super(SimpleGCNNet, self).__init__()
        self.pooling = pooling
        self.gnn = SimpleGCN(in_feats,hid_feats,out_feats,pooling)

        if (self.pooling == 'mean_max') or (self.pooling=='scatter_mean_max'):
          self.fc1 = nn.Linear(out_feats+out_feats,D_H)
        else:
          self.fc1 = nn.Linear(out_feats,D_H)
        self.fc2 = nn.Linear(D_H,D_out)

    def forward(self, data):
        x = self.gnn(data)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x



class TripleGCN(nn.Module):
    """default pooling set as scatter_mean with dropout rate as 0.5"""
    def __init__(self,in_feats,hid_feats,out_feats,pooling='scatter_mean',dropout=0.5):
        super(TripleGCN, self).__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.conv1 = GraphConv(in_feats, hid_feats*2)
        self.conv2 = GraphConv(hid_feats*2, hid_feats)
        self.conv3 = GraphConv(hid_feats,out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout+0.1, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        if self.pooling == 'scatter_mean':
          x = scatter_mean(x,data.batch,dim=0)
        elif self.pooling == 'scatter_max':
          x = scatter_max(x,data.batch,dim=0)
        elif self.pooling == 'scatter_add':
          x = scatter_add(x,data.batch,dim=0)
        elif self.pooling == 'global_mean':
          x = nn.global_mean_pool(x,data.batch)
        elif self.pooling == 'global_max':
          x = nn.global_max_pool(x,data.batch)
        elif self.pooling == 'mean_max':
          x_mean = nn.global_mean_pool(x,data.batch)
          x_max = nn.global_max_pool(x,data.batch)
          x = torch.cat((x_mean,x_max), 1)
        elif self.pooling == 'scatter_mean_max':
          x_mean = scatter_mean(x,data.batch,dim=0)
          x_max = scatter_add(x,data.batch,dim=0)
          x = torch.cat([x_mean,x_max],1)
        elif self.pooling == 'root':
          rootindex = data.rootindex
          root_extend = torch.zeros(len(data.batch), 768).to(device)
          batch_size = max(data.batch) + 1
          for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x[rootindex[num_batch]]
          x = root_extend
        else:
          assert False, "Something wrong with the parameter --pooling"

        return x



class TripleGCNNet(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,D_H, D_out,pooling='scatter_mean'):
        super(TripleGCNNet, self).__init__()
        self.pooling = pooling
        self.gnn = TripleGCN(in_feats,hid_feats,out_feats,pooling='scatter_mean')
        if (self.pooling == 'mean_max'):
          self.fc1 = nn.Linear(out_feats+out_feats,D_H)
        else:
          self.fc1 = nn.Linear(out_feats,D_H)
        self.fc2 = nn.Linear(D_H,D_out)

    def forward(self, data):
        x = self.gnn(data)
        emb = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return emb, x
