import torch
from torch import nn
import torchvision.transforms as T

import utils
from agent.networks.encoder import Encoder

class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, hidden_dim):
		super().__init__()

		self._output_dim = action_shape[0]
		
		self.policy = nn.Sequential(nn.Linear(repr_dim*2, hidden_dim),
									nn.ReLU(),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(),
									nn.Linear(hidden_dim, self._output_dim))

		self.apply(utils.weight_init)

	def forward(self, obs, std):
		mu = torch.tanh(self.policy(obs))
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)
		return dist

class BCAgent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, stddev_schedule, 
	      		 stddev_clip, use_tb, obs_type):
		self.device = device
		self.lr = lr
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_tb = use_tb
		self.use_encoder = True if obs_type=='pixels' else False
		
		# actor parameters
		self._act_dim = action_shape[0]

		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
		else:
			repr_dim = obs_shape[0]

		self.actor = Actor(repr_dim, action_shape, hidden_dim).to(device)

		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters())
		self.actor_opt = torch.optim.Adam(self.actor.parameters())

		# data augmentation
		if self.use_encoder:
			self.aug = utils.RandomShiftsAug(pad=4)

		self.train()

	def __repr__(self):
		return "gcbc"
	
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

	def act(self, obs, goal, step):
		# convert to tensor and add batch dimension
		obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
		goal = torch.as_tensor(goal, device=self.device).float().unsqueeze(0)

		stddev = utils.schedule(self.stddev_schedule, step)

		if self.use_encoder:
			obs=self.encoder(obs)
		obs_goal = torch.cat([obs, goal], dim=-1)
		dist_action = self.actor(obs_goal, stddev)		
		action = dist_action.mean
		return action.cpu().detach().numpy()[0]

	def update(self, expert_replay_iter, step):
		metrics = dict()

		batch = next(expert_replay_iter)
		obs, action, goal = utils.to_torch(batch, self.device)
		obs, action, goal = obs.float(), action.float(), goal.float()

		# augment
		if self.use_encoder:
			obs = self.aug(obs)
			obs = self.encoder(obs)
		obs_goal = torch.cat([obs, goal], dim=-1)

		stddev = utils.schedule(self.stddev_schedule, step)
		
		dist_action = self.actor(obs_goal, stddev)
		log_prob = dist_action.log_prob(action).sum(-1, keepdim=True)
		actor_loss = -log_prob.mean()

		if self.use_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.actor_opt.zero_grad(set_to_none=True)  
		actor_loss.backward()
		if self.use_encoder: 
			self.encoder_opt.step()
		self.actor_opt.step()
  
		# log
		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
		
		return metrics

	def save_snapshot(self):
		keys_to_save = ['actor']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v
