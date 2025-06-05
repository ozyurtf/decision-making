import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
import pickle

class ExpertReplayBuffer(IterableDataset):
    def __init__(self, dataset_path, obs_type, height, width, train_test_ratio):
        """
        Initializing the goal and sequence of observations - actions 
        for each episode along with the maximum length of the episodes and
        the number of episodes that will be used to train a model. 
        """
        self._height = height
        self._width = width
        self._train_test_ratio = train_test_ratio
        
        with open(dataset_path, 'rb') as f:
            if obs_type == 'pixels':
                obses, _, actions, _, goals = pickle.load(f)
            elif obs_type == 'features':
                _, obses, actions, _, goals = pickle.load(f)
        
        self._episodes = []
        for i in range(len(obses)):
            episode = dict(observation=obses[i],
                           action=actions[i],
                           goal=goals[i])
            self._episodes.append(episode)
            
        self._max_episode_len = max([len(episode) for episode in self._episodes])
        self._train_episodes_till = int(len(self._episodes) * self._train_test_ratio)

    def _sample(self):
        """
        Sampling an episode, sampling an observation and action 
        at a random timestep within that episode, normalizing 
        the observation and the goal, and returning 
        the sampled observation, sampled action, and 
        the corresponding goal of the sampled episode. 
        """
        # Sample an episode.
        episode = random.choice(self._episodes[:self._train_episodes_till])
        
        # List of observations in the sampled episode. 
        observations = episode['observation'] 
        
        # List of actions in the sampled episode. 
        actions = episode['action']
        
        # The goal of the episode.
        goal = np.array(episode['goal'])

        # Sample an observation and action 
        # from the list of observations and the list of actions
        # that belong to the sampled episode.
        sample_idx = np.random.randint(0, len(observations))
        sampled_obs = observations[sample_idx]
        sampled_action = actions[sample_idx]
        
        # Assign the goal of the sampled episode to a variable.
        goal = goal[sample_idx]

        # Normalize the sampled observation.
        sampled_obs = np.array(sampled_obs)
        sampled_obs[0] = sampled_obs[0] / self._height
        sampled_obs[1] = sampled_obs[1] / self._width
        
        # Normalize the goal of the sampled episode.
        goal = np.array(goal)
        goal[0] = goal[0] / self._height
        goal[1] = goal[1] / self._width

        return (sampled_obs, sampled_action, goal)
    
    def sample_test(self):
        """
        Repeating the same process as in the _sample() function 
        -— except now we sample from the list of episodes 
        that will be used to test the model.
        """
        episode = random.choice(self._episodes[self._train_episodes_till:])
        observation = episode['observation']
        goal = np.array(episode['goal'])
        
        start_obs = observation[0]
        start_obs = np.array(start_obs)
        start_obs[0] = start_obs[0] / self._height
        start_obs[1] = start_obs[1] / self._width

        goal = np.array(goal)
        goal[:,0] = goal[:,0] / self._height
        goal[:,1] = goal[:,1] / self._width
        
        return (start_obs, goal)

    def __iter__(self):
        """
        Sampling data continuously during training.
        """
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    """
    This is used to ensure that each worker 
    samples a different observation, action, and goal
    to avoid duplicate data during the training process.
    """
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

def make_expert_replay_loader(replay_dir, batch_size, obs_type, height, width, train_test_ratio):
    """
    Wrapping around our dataset to iterate over it in batches during training.
    In each iteration, each worker (thread) 
    
    - samples an episode, 
    - samples an observation and action at a random timestep within that episode, 
    - returns them along with the goal of the sampled episodes,
    - stores the data in a memory region that allows for faster transfers between CPU and GPU memory
    (this makes the training process much faster because 
    GPU can access this memory region with Direct Memory Access
    without CPU and this eliminates the process of copying data from CPU to GPU)
    
    independently. The data stored in the memory region 
    is collected into batches and the entire process continues as 
    the DataLoader iterates.
    """
    iterable = ExpertReplayBuffer(replay_dir, obs_type, height, width, train_test_ratio)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=2,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader