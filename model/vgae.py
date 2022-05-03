import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.distributions import Bernoulli

from layers import GraphConvolution
pyro.enable_validation(True)
from torch_geometric.nn import GCNConv

# ------------------------------------
# Some functions borrowed from:
# https://github.com/vmasrani/gae_in_pytorch/blob/master/models.py
# with modifications
# ------------------------------------


class GCNEncoder(nn.Module):
	def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
		super(GCNEncoder, self).__init__()
		self.gc1 = GCNConv(input_feat_dim, hidden_dim1, dropout, act=F.relu)
		self.gc2 = GCNConv(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
		self.gc3 = GCNConv(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
		self.dc = InnerProductDecoder(dropout, act=lambda x: x)

	def encode(self, x, adj):
		hidden1 = self.gc1(x, adj)
		return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

	def reparameterize(self, mu, logvar):
		if self.training:
			std = torch.exp(logvar)
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)
		else:
			return mu

	def forward(self, x, adj):
		mu, logvar = self.encode(x, adj)
		z = self.reparameterize(mu, logvar)
		return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
	"""Decoder for using inner product for prediction."""

	def __init__(self, dropout, act=torch.sigmoid):
		super(InnerProductDecoder, self).__init__()
		self.dropout = dropout
		self.act = act

	def forward(self, z):
		z = F.dropout(z, self.dropout, training=self.training)
		adj = self.act(torch.mm(z, z.t()))
		return adj

class UserGAE(nn.Module):
	"""Graph Auto Encoder (see: https://arxiv.org/abs/1611.07308)"""

	def __init__(self, data, n_hidden, n_latent, dropout, subsampling=False):
		super(UserGAE, self).__init__()

		# Input Data
		self.uid = data['uid']
		self.x = data['features']
		self.adj_norm = data['adj_norm']
		self.adj_labels = data['adj_labels']
		self.obs = self.adj_labels.view(1, -1)

		# Dimensions
		N, D = data['features'].shape
		self.n_samples = N
		self.n_edges = self.adj_labels.sum()
		self.n_subsample = 2 * self.n_edges
		self.input_dim = D
		self.n_hidden = n_hidden
		self.n_latent = n_latent

		# Parameters
		self.pos_weight = float(N * N - self.n_edges) / self.n_edges
		self.norm = float(N * N) / ((N * N - self.n_edges) * 2)
		self.subsampling = subsampling

		# Layers
		self.dropout = dropout
		self.encoder = GCNEncoder(self.input_dim, self.n_hidden, self.n_latent, self.dropout)
		self.decoder = InnerProductDecoder(self.dropout)


	def model(self):
		# register PyTorch module `decoder` with Pyro
		pyro.module("decoder", self.decoder)

		# Setup hyperparameters for prior p(z)
		z_mu    = torch.zeros([self.n_samples, self.n_latent])
		z_sigma = torch.ones([self.n_samples, self.n_latent])

		# sample from prior
		z = pyro.sample("latent", dist.Normal(z_mu, z_sigma).to_event(2))

		# decode the latent code z
		z_adj = self.decoder(z).view(1, -1)

		# Score against data, removed the weighted one
		pyro.sample('obs', Bernoulli(z_adj, weight=self.pos_weight).to_event(2), obs=self.obs)


	def guide(self):
		# register PyTorch model 'encoder' w/ pyro
		pyro.module("encoder", self.encoder)

		# Use the encoder to get the parameters use to define q(z|x)
		z_mu, z_sigma = self.encoder(self.x, self.adj_norm)

		# Sample the latent code z
		pyro.sample("latent", dist.Normal(z_mu, z_sigma).to_event(2))


	def get_embeddings(self):
		z_mu, _ = self.encoder.eval()(self.x, self.adj_norm)
		# Put encoder back into training mode
		self.encoder.train()
		return z_mu
