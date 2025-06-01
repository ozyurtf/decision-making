import torch
from torch import nn

import torchvision.transforms as T
from torch.nn import functional as F
import torch.distributions as D

import utils
from agent.networks.encoder import Encoder
from agent.networks.kmeans_discretizer import KMeansDiscretizer

class FocalLoss(nn.Module):
	
	def __init__(self, gamma: float = 0, size_average: bool = True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.size_average = size_average

	def forward(self, input, target):
		"""
		Args:
			input: (N, B), where B = number of bins
			target: (N, )
		"""
		logpt = F.log_softmax(input, dim=-1)
		logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
		pt = logpt.exp()

		loss = -1 * (1 - pt) ** self.gamma * logpt
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()

class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, hidden_dim, nbins):
		super().__init__()

		self._output_dim = action_shape[0]

		self.nbins = nbins
		self.policy = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
									nn.ReLU(),
									nn.Linear(hidden_dim, hidden_dim),
                      				nn.ReLU()) 
  
		self.binning_head = nn.Sequential(nn.Linear(hidden_dim, nbins))
		self.offset_head = nn.Sequential(nn.Linear(hidden_dim, self._output_dim*nbins),
                                   		 nn.Tanh())
  
		self.apply(utils.weight_init)

	def forward(self, obs, std, cluster_centers=None):
		obs_features = self.policy(obs)
		bin_logits = self.binning_head(obs_features)
		offsets = self.offset_head(obs_features)
		offsets = offsets.view(-1, self.nbins, self._output_dim)
		return bin_logits, offsets 

class Agent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, stddev_schedule, 
	      		 stddev_clip, use_tb, obs_type, nbins, kmeans_iters, offset_weight,
				 offset_loss_weight):
		self.device = device
		self.lr = lr
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_tb = use_tb
		self.use_encoder = True if obs_type == 'pixels' else False
		self.nbins = nbins
		self.kmeans_iters = kmeans_iters
		self.offset_weight = offset_weight
		self.offset_loss_weight = offset_loss_weight
		
		# actor parameters
		self._act_dim = action_shape[0]

		# discretizer
		self.discretizer = KMeansDiscretizer(num_bins = self.nbins, kmeans_iters = self.kmeans_iters)

		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
		else:
			repr_dim = obs_shape[0]

		self.actor = Actor(repr_dim, action_shape, hidden_dim, nbins).to(device)

		# Loss
		self.criterion = FocalLoss(gamma=2.0)
		self.mse_loss = nn.MSELoss()

		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr = 0.01)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr = 0.01)

		# Data augmentation
		if self.use_encoder:
			self.aug = utils.RandomShiftsAug(pad=4)

		self.train()

	def __repr__(self):
		return "bet"
	
	def train(self, training=True):
		self.training = training
		if training:
			if self.use_encoder:
				self.encoder.train(training)
			self.actor.train(training)
		else:
			if self.use_encoder:
				self.encoder.eval()
			self.actor.eval()
	
	def compute_action_bins(self, actions):
		# Compute nbins bin centers using k-nearest neighbors algorithm 
		actions = torch.as_tensor(actions, device=self.device).float()
		self.discretizer.fit(actions)
		self.cluster_centers = self.discretizer.bin_centers.float().to(self.device)
		
	def find_closest_cluster(self, actions) -> torch.Tensor:
		distances = torch.cdist(actions, self.cluster_centers)  # (N, num_bins)
		return torch.argmin(distances, dim=1)  # (N, )
 
	def act(self, obs, goal, step):
		obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
		goal = torch.as_tensor(goal, device=self.device).float().unsqueeze(0)
		
		stddev = utils.schedule(self.stddev_schedule, step)
		if self.use_encoder: 
			obs=self.encoder(obs)
		bin_logits, offsets = self.actor(obs, stddev, self.cluster_centers)

		bin_probabilities = F.softmax(bin_logits, dim = -1)
		sampled_bin = torch.argmax(bin_probabilities, dim = -1) 
		base_action = self.cluster_centers[sampled_bin]
		max_indices = bin_logits.argmax(dim=-1)
		offset = offsets[torch.arange(offsets.size(0)), max_indices]
		action = base_action + offset
		return action.cpu().numpy()[0]

	def update(self, expert_replay_iter, step):
		metrics = dict()

		batch = next(expert_replay_iter)
		obs, action, goal = utils.to_torch(batch, self.device)
		obs, action, goal = obs.float(), action.float(), goal.float()
		
		# augment
		if self.use_encoder:
			obs = self.aug(obs)
			obs = self.encoder(obs)

		stddev = utils.schedule(self.stddev_schedule, step)
		bin_logits, offsets = self.actor(obs, stddev, self.cluster_centers)

		action_closest_cluster_centers = self.find_closest_cluster(action) # action_closest_cluster_centers = ⌊a⌋, action = a, cluster centers = A
		discrete_loss = self.criterion(bin_logits, action_closest_cluster_centers)  	
  	
		residual_action = action - self.cluster_centers[action_closest_cluster_centers] # ⟨a⟩
  
		max_indices = bin_logits.argmax(dim=-1)
		offset = offsets[torch.arange(offsets.size(0)), max_indices]
    
		offset_loss = self.mse_loss(residual_action, offset)

		# actor loss
		actor_loss = discrete_loss + self.offset_loss_weight * offset_loss
		
		if self.use_encoder: 
			self.encoder_opt.zero_grad()
		self.actor_opt.zero_grad()
  
		actor_loss.backward()

		if self.use_encoder: 
			self.encoder_opt.step()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['discrete_loss'] = discrete_loss.mean().item()
			metrics['offset_loss'] = offset_loss.mean().item() * self.offset_loss_weight
			metrics['logits_entropy'] = D.Categorical(logits=bin_logits).entropy().mean().item()

		return metrics

	def save_snapshot(self):
		keys_to_save = ['actor', 'cluster_centers']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v

		# Update optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
