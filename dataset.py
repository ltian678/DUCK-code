from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from utils import preprocessing_for_bert_latest, preprocessing_for_bert_seq





class TextUserData(Data):
	def __init__(self, text_x, user_x, text_edge_index, user_edge_index,y,idx):
		super(PairData, self).__init__()
		self.text_x = text_x
		self.user_x = user_x
		self.text_edge_index = text_edge_index
		self.user_edge_index = user_edge_index
		self.y = y
		self.idx = idx

	def __inc__(self, key, value):
		if key == 'text_edge_index':
				return self.text_x.size(0)
		if key == 'user_edge_index':
				return self.user_x.size(0)
		else:
				return super().__inc__(key, value)



class DUCKData(Data):
	def __init__(self, text_x, user_x, seq_x, text_edge_index, user_edge_index,y,idx):
		super(PairData, self).__init__()
		self.text_x = text_x
		self.user_x = user_x
		self.seq_x = seq_x
		self.text_edge_index = text_edge_index
		self.user_edge_index = user_edge_index
		self.y = y
		self.idx = idx

	def __inc__(self, key, value):
		if key == 'text_edge_index':
				return self.text_x.size(0)
		if key == 'user_edge_index':
				return self.user_x.size(0)
		else:
				return super().__inc__(key, value)


class CommentTreeDataset(Dataset):
	def __init__(self,fold_x,data_path):
		self.fold_x = fold_x
		self.data_path = data_path

	def __len__(self):
		return len(self.fold_x)

	def __getitem__(self,index):
		id = self.fold_x[index]
		data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
		idx = int(id)
		str_idx = str(idx)
		input_ids, attention_mask = preprocessing_for_bert_latest(data['root'],data['nodecontent']) #convert list of strings to list of input_ids and attention_mask for this idx
		input_ids_seq, attention_mask_seq = preprocessing_for_bert_seq(data['root'],data['nodecontent'])

		return Data(
				edge_index = torch.LongTensor(data['edgematrix']),
				root = torch.LongTensor(data['root']),
				y = torch.LongTensor([int(data['y'])]),
				rootindex = torch.LongTensor([int(data['rootindex'])]),
				idx = torch.LongTensor([int(idx)]),
				input_ids = torch.LongTensor(input_ids),
				attention_mask = torch.LongTensor(attention_mask),
				input_ids_seq = torch.LongTensor(input_ids_seq),
				attention_mask_seq = torch.LongTensor(attention_mask_seq),
				top_index = torch.LongTensor(data['topindex']),
				tri_index = torch.LongTensor(data['triIndex'])),

def collate_fn(data):
	return data


class UserTreeDataset(Dataset):
	def __init__(self,fold_x,data_path):
		self.fold_x = fold_x
		self.data_path = data_path

	def __len__(self):
		return len(self.fold_x)

	def __getitem__(self,index):
		id = self.fold_x[index]
		data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
		idx = int(id)
		return Data(
				x = torch.tensor(data['root'],dtype=torch.float32),
				edge_index = torch.LongTensor(data['edgematrix']),
				y = torch.LongTensor([int(data['y'])]),
				idx = torch.LongTensor([int(idx)]),
		)


def collate_fn(data):
	return data




class DuckDataset(Dataset):
	def __init__(self, fold_x, data_path):
		self.fold_x = fold_x
		self.data_path = data_path

	def __len__(self):
		return len(self.fold_x)

	def __getitem__(self, index):
		id = self.fold_x[index]
		data = np.load(os.path.join(self.data_path, id + '.npz'), allow_pickle=True)
		idx = int(id)
		user_x = torch.tensor(data['userx'],dtype=torch.float32)
		user_edge_index = torch.LongTensor(data['useredgematrix'])
		text_x = torch.tensor(data['textx'],dtype=torch.float32)
		text_edge_index = torch.LongTensor(data['textedgematrix'])
		seq_x = torch.tensor(data['seqx'],dtype=torch.float32)

		y = torch.LongTensor([int(data['y'])])
		idx = torch.LongTensor([int(idx)])
		duck_data = DUCKData(text_x=text_x, user_x=user_x, seq_x=seq_x, text_edge_index=text_edge_index, user_edge_index=user_edge_index,y=y,idx=idx)
		return duck_data

def collate_fn(data):
	return data
