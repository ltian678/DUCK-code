import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv,GraphConv,GINConv,GATConv
import copy
from torch_scatter import scatter_mean, scatter_max, scatter_add



class SimpleGAT(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,pooling='scatter_mean',dropout=0.6):
        super(SimpleGAT, self).__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.conv1 = GATConv(in_feats, hid_feats, heads=8,dropout=dropout)
        self.conv2 = GATConv(hid_feats*8, out_feats,heads=8,concat=False,dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
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



class SimpleGATNet(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,res_feats,pooling='scatter_mean',dropout):
        super(SimpleGATNet, self).__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.gnn = SimpleGAT(in_feats, hid_feats, out_feats,pooling,dropout)
        if (self.pooling == 'mean_max') or (self.pooling=='scatter_mean_max'):
          self.fc = nn.Linear(out_feats+out_feats,res_feats)
        else:
          self.fc = nn.Linear(out_feats,res_feats)


    def forward(self, data):
        x = self.gnn(data)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x



class TripleGAT(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,pooling='scatter_mean',dropout):
        super(TripleGAT, self).__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.conv1 = GATConv(in_feats, hid_feats*2, heads=8,dropout=dropout)
        self.conv2 = GATConv(hid_feats*8*2, hid_feats,heads=8,dropout=dropout)
        self.conv3 = GATConv(hid_feats*8, out_feats,heads=1,concat=False,dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
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



class TripleGATNet(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,res_feats,pooling='scatter_mean',dropout):
        super(TripleGATNet, self).__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.gnn = TripleGAT(in_feats, hid_feats, out_feats,pooling,dropout)
        if (self.pooling == 'mean_max') or (self.pooling=='scatter_mean_max'):
          self.fc = nn.Linear(out_feats+out_feats,res_feats)
        else:
          self.fc = nn.Linear(out_feats,res_feats)


    def forward(self, data):
        x = self.gnn(data)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
