import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv,GraphConv,GINConv,GATConv
import copy
from torch_scatter import scatter_mean, scatter_max, scatter_add

class SimpleGAT_BERT(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,n_heads,gat_dropout,pooling='scatter_mean'):
        super(SimpleGAT_BERT, self).__init__()
        self.pooling = pooling
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1 = GATConv(in_feats, hid_feats, heads=n_heads,dropout=gat_dropout)
        self.conv2 = GATConv(hid_feats*n_heads, out_feats,heads=n_heads,concat=False,dropout=gat_dropout)

    def forward(self, data):
        #x, edge_index = data.x, data.edge_index
        input_ids, attention_mask = data.input_ids, data.attention_mask
        batch_size = max(data.batch) + 1
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        x, edge_index = last_hidden_state_cls, data.edge_index
        #print('*******************After  x.shape', x.shape)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
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
          print('x_mean',type(x_mean))
          print('x_mean shape',x_mean.shape)
          print('x_max shape', x_max.shape)
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



class SimpleGATBERTNet(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,D_in,H,D_out,pooling='scatter_mean'):
        super(SimpleGATBERTNet, self).__init__()
        self.pooling = pooling
        #D_in, H = 768,32,4
        self.gnn = SimpleGAT_BERT(D_in,hid_feats,out_feats,pooling)

        if (self.pooling == 'mean_max') or (self.pooling=='scatter_mean_max'):
          self.fc1 = nn.Linear(out_feats+out_feats,H)
        else:
          self.fc1 = nn.Linear(out_feats,H)
        self.fc2 = nn.Linear(H,D_out)


    def forward(self, data):
        gnn_x = self.gnn(data)
        #print('x.shape',x.shape)
        x = self.fc1(gnn_x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x



class TripleGAT_BERT(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,pooling='scatter_mean'):
        super(TripleGAT_BERT, self).__init__()
        self.pooling = pooling
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1 = GATConv(in_feats, hid_feats*2, heads=8,dropout=0.6)
        self.conv2 = GATConv(hid_feats*8*2, hid_feats,heads=8,dropout=0.6)
        self.conv3 = GATConv(hid_feats*8, out_feats,heads=1,concat=False,dropout=0.6)

    def forward(self, data):
        #x, edge_index = data.x, data.edge_index
        input_ids, attention_mask = data.input_ids, data.attention_mask
        batch_size = max(data.batch) + 1
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        x, edge_index = last_hidden_state_cls, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
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
          x_max = scatter_max(x,data.batch,dim=0)
          x = torch.cat([x_mean,x_max],1)
        elif self.pooling == 'root':
          rootindex = data.rootindex
          root_extend = torch.zeros(len(data.batch), 5000).to(device)
          batch_size = max(data.batch) + 1
          for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x[rootindex[num_batch]]
          x = root_extend
        else:
          assert False, "Something wrong with the parameter --pooling"
        return x



class TripleGATBERTNet(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,D_in,H,D_out,pooling='scatter_mean'):
        super(TripleGATBERTNet, self).__init__()
        self.pooling = pooling
        self.gnn = TripleGAT_BERT(in_feats, hid_feats, out_feats,pooling)

        self.fc = nn.Linear(out_feats,D_out)


    def forward(self, data):
        x = self.gnn(data)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
