import torch
from torch import nn
import torchvision.transforms as T
import os
import utils
from agent.networks.encoder import Encoder

# Here we define a neural network that will be used to take 
# the pixels of the scene that are encoded or 
# features that represent the environment as input 
# and produces an action. This model is named Actor. 
# During the forward propagation, we put 
# the encoded version of the observation as input, and 
# predict a probability distribution in the form of gaussian. 
# By taking the mean of this probability distribution, 
# we can predict the best action. 
class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, hidden_dim):
		super().__init__()

		self._output_dim = action_shape[0]
		
		# Define the policy network
		self.policy = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
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

# Here we initialize the actor model that will be used to take an action 
# in the environment, encoder model that will be used to encode the 
# pixels, optimizers that will be used to update the parameters of these models, 
# training details (e.g., learning rate, device, etc.), data augmentation method 
# that we will apply to the pixels, etc. 
 
# We define a function named "train" to train and evaluate
# the actor and encoder models. 

# We define another function named "act" that will get 
# observation, goal, and the current step, predict a probability distribution, 
# and return the mean of this distribution since the mean represents 
# the most likely action. 

# We define another function named "update" that samples an episode. 
# An episode contains goal(s) and a sequence of observations and activations 
# that belong to an expert. Once we obtain an episode, 
# if the observation is in the form of pixels, 
# we first augment the observations and then encode the observations with CNN model. 
# Then we put the (encoded) observations, actions, and goal(s) into the 
# action model to get the mean of the probability distribution of the possible actions 
# that was predicted by the actor model as the representation of the best action. 
# Lastly, we measure the error between the predicted best action and real action 
# and update the parameters of the actor and encoder models. 
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

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
		else:
			repr_dim = obs_shape[0]

		self.actor = Actor(repr_dim, action_shape, hidden_dim).to(device)

		# TODO: Define optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters())
		self.actor_opt = torch.optim.Adam(self.actor.parameters())

		# data augmentation
		if self.use_encoder:
			self.aug = utils.RandomShiftsAug(pad=4)

		self.train()

	def __repr__(self):
		return "bc"
	
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
		obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0) # 1 x 2
		goal = torch.as_tensor(goal, device=self.device).float().unsqueeze(0) # 1 x 2
		stddev = utils.schedule(self.stddev_schedule, step)
  
		# TODO: Compute action using the actor (and the encoder if pixels are used)
		if self.use_encoder:
			obs=self.encoder(obs)
		dist_action = self.actor(obs, stddev)		
		action = dist_action.mean
		return action.cpu().detach().numpy()[0]

	def update(self, expert_replay_iter, step):
		metrics = dict()

		batch = next(expert_replay_iter)
		obs, action, goal = utils.to_torch(batch, self.device)
		obs, action, goal = obs.float(), action.float(), goal.float() # obs = 256 x 2, action = 256 x 2, goal = 256 x 2
		
		# augment
		if self.use_encoder:
			# TODO: Augment the observations and encode them (for pixels)
			obs = self.aug(obs)
			obs = self.encoder(obs)

		stddev = utils.schedule(self.stddev_schedule, step)
		
		# TODO: Compute the actor loss using log_prob on output of the actor
		dist_action = self.actor(obs, stddev)
		log_prob = dist_action.log_prob(action).sum(-1, keepdim=True)
		actor_loss = -log_prob.mean()
		 
		# TODO: Update the actor (and encoder for pixels)	
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
