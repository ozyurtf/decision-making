import gym
from gym import spaces
import cv2
import numpy as np

class ParticleEnv(gym.Env):
    def __init__(self, height=84, width=84, step_size=10, reward_type='dense', 
                   reward_scale=None, block=None, fixed_goal=False, num_goals=1):
        super(ParticleEnv, self).__init__()

        # Define variables that will be used to
        # - create an observation space 
        # - create an action space
        # - initialize the agent to a random location 
        # - create a random goal 
        # - determine the reward amount after taking actions 
        # - put obstacled to the environment
        self.height = height
        self.width = width
        self.step_size = step_size
        self.reward_type = reward_type
        self.block = block
  
        if reward_scale is None: 
            self.reward_scale = np.sqrt(height**2 + width**2)
        elif reward_scale is not None: 
            self.reward_scale = reward_scale  

        # Initialize an empty list to store all goal positions in the environment.
        # The environment code supports the agent to reach multiple goals in sequence.
        self.goals = []
        
        # Initialize a boolean value that determines whether the goal stays constant 
        # across different episodes. If this is initialized as True, this means that 
        # goals will be in the same position when the environment is reset.
        # Otherwise, a new goal position is generated whenever the environment is reset.
        self.fixed_goal = fixed_goal
        
        # Specify the number of goals the agent needs to reach.
        self.num_goals = num_goals
        
        # Create a 2 dimensional observation space in the form of a box
        # that starts from (0,0) at top left and 
        # reaches to the (height - 1, width -1) at bottom right.        
        self.observation_space = spaces.Box(low = np.array([0,0],dtype=np.float32), 
                                            high = np.array([self.height-1, self.width-1],dtype=np.float32),
                                            dtype = np.float32)
        

        # Create an action space in which an action can make the agent to move 
        # anywhere between -step size and +step size units in both x and y directions.         
        self.action_space = spaces.Box(low = np.array([-step_size, -step_size],dtype=np.float32), 
                                       high = np.array([step_size, step_size],dtype=np.float32),
                                       dtype = np.float32)
    
    def step(self, action):
        # Store the current state in 'prev_state' before moving to the next state
        prev_state = self.state
        
        # Update the state based on the action taken and step size. 
        # The action is multiplied by the step size to determine how far 
        # away the agent should go from the current state.        
        self.state = np.array([self.state[0] + self.step_size * action[0], 
                               self.state[1] + self.step_size * action[1]], dtype=np.float32)
        
        
        # If the agent moves outside the environment and if the reward type is dense, 
        # in other words if the agent receives feedbacks frequently instead of receiving 
        # only after completing a task, give negative reward (-10) to the agent. 
        # If the reward type is sparse, in other words, the agent receives feedbacks 
        # only after completing a task, don't give any positive or negative reward. 
        # Then take the agent to the previous state which was within the environment.        
        if self.state[0] < 0 or self.state[0] >= self.height or self.state[1]<0 or self.state[1] >= self.width:
            if self.reward_type == 'dense': 
                reward = -10 
            else: 
                reward = 0
            self.state = prev_state
            done = False
            
        # If the agent hits an obstacle and if the reward type is dense, 
        # give negative reward (-10) to the agent. If the reward type is sparse
        # don't give any positive or negative reward. 
        # Then take the agnet to the previous state which was within the environment.
        elif self.observation[int(self.state[0]), int(self.state[1])]==1:
            if self.reward_type == 'dense': 
                reward = -10 
            else: 
                reward = 0
            self.state = prev_state
            done = False
        
        # If the agent reaches one of the goals and the goal 
        # it reached is the correct goal in the sequence, 
        # increment the count of the number of the goals reached
        # and give the agent a positive reward of 1. 
        
        # If the agent reaches one of the goals but the goal 
        # it reached is not the correct goal in the sequence, 
        # give the reward of -10 to the agent if the reward type is dense 
        # and don't give any positive or negative reward if the reward type is sparse.
        
        # Finally, if the agent reached all the goals,
        # set the done variable as True to stop the agent to take further actions. 
        # Otherwise, set the done variable as False so that the agent can 
        # try to reach the remaining goals.
        elif self.observation[int(self.state[0]), int(self.state[1])] >= 2:
            if self.observation[int(self.state[0]),
                                int(self.state[1])] == self.num_goals_reached + 2:
                self.num_goals_reached += 1
                reward = 1 
            else:
                if self.reward_type == 'dense': 
                    reward = -10 
                else: 
                    reward = 0
            if self.num_goals_reached == self.num_goals: 
                done = True 
            else: 
                done = False

        # If these are not the cases, this means that 
        # the agent did not hit an obstacle, did not reach all the goals, 
        # and did not move outside the environment 
        # after taking the action. This means that the action is valid and therefore 
        # we don't give any positive or negative reward to the agent 
        # and don't end the episode.
        
        # One alternative option of setting the reward is 
        # calculating the distance between the agent and the final goal,
        # normalize this distance by dividing it by reward scale, 
        # make it negative, and assign this value as the reward to the agent so that 
        # being farther away from the final goal results in a larger penalty 
        # for the agent. 
        else:
            # if self.reward_type == 'sparse': 
            #     reward = -1
            # elif self.reward_type == 'dense': 
            #     reward = -np.sqrt((self.goals[-1][0]-self.state[0])**2 +
            #                       (self.goals[-1][0]-self.state[1])**2) / self.reward_scale            
            reward = 0
            done = False
            
        # Increment the step number.    
        self._step += 1
        
        # Define the success. If the reward type is dense, 
        # the agent is seen as successful by making a valid action
        # without reaching a goal, hitting an obstacle, and 
        # moving out of the environment.
        # If the reward type is sparse, the agent is seen as successful
        # if it doesn't reach the goal after taking the action.        
        info = {}
        if reward == 0: 
            info['is_success'] = 1
        else: 
            info['is_success'] = 0
            
        # Normalize the state of the agent after taking the action 
        # and store it in an array.
        state = np.array([self.state[0] / self.height, self.state[1] / 
                          self.width]).astype(np.float32)

        # Return the (normalized) state and the reward that is received 
        # after taking the action along with the information of 
        # whether the agent reached the goal and if it was successful 
        # after taking the action.
        return state, reward, done, info
    
    def reset(self, start_state=None, reset_goal=False, goal_state=None, demo=None):
        # If two dimensional start state is not None, 
        # convert the normalized state back to its original scale,
        # convert it into array, and make the two variables that represent 
        # the initial state of the agent int32.
        if start_state is not None:
            start_state[0] = start_state[0] * self.height
            start_state[1] = start_state[1] * self.width
            self.state = np.array(start_state).astype(np.int32)        

        # If two dimensional start state is None, 
        # assign the agent to a random location within the environment.
        elif start_state is None:
            self.state = np.array([np.random.randint(0, self.height), 
                                   np.random.randint(0, self.width)]).astype(np.int32)

        if len(self.goals) == 0:
            # If the total number of the goals in the environment is 0 and
            # if two dimensional location of the goal_state expressed in the reset() function
            # is None, assign a random goal position and ensure that the first location of the agent
            # (start state) is not the same as this randomly initialized goal position.
            # After this, add the randomly initialized goal position to the list of goals.             
            if goal_state is None:
                goal = np.array([np.random.randint(0, self.height), 
                                 np.random.randint(0, self.width)]).astype(np.int32)
                while (self.state == goal).all():
                    goal = np.array([np.random.randint(0, self.height), 
                                     np.random.randint(0, self.width)]).astype(np.int32)
                self.goals.append(goal)
                                    
            elif goal_state is not None:
                # If the total number of the goals in the environment is 1 or
                # there is no demonstration (a recorded trajectory of how an expert
                # navigated through the environment to reach a goal,
                # convert the (normalized) goal back to its original scale
                # and append it to the list of goals. 
                # Even if he number of goals to reach is more than 1,
                # if there is no demonstration (demo),
                # this means that we cannot use the demo
                # to create intermediate goals. So, 
                # we choose a simpler single-goal behavior by default.
                if self.num_goals == 1 or demo is None:
                    goal_state[0] = goal_state[0] * self.height
                    goal_state[1] = goal_state[1] * self.width
                    self.goals.append(np.array(goal_state).astype(np.int32))
                    
                # If the total number of the goals in the environment is more than 1
                # and a demonstration that shows a recorded trajectory of how 
                # and expert navitages through the environment exists,
                # calculate the positions of the intermediate goals in the demonstration, 
                # convert the coordinates of the positions of the intermediate goals 
                # back to the original scale and append them to a list of goals. 
                
                # In addiiton, if the number of goals is more than one, 
                # we treat the goal_state expressed as the parameter of the 
                # reset() function as the final goal, convert the coordiantes 
                # of it back to the original scale and append these coordiantes 
                # to the list of goals. This will be the final goal of the agent.
                elif self.num_goals > 1:
                    
                    intermediate_goal_idx = []
                    for i in range(self.num_goals - 1): 
                        goal_idx = (i + 1) * len(demo) // self.num_goals
                        intermediate_goal_idx.append(goal_idx)    
                    
                        intermediate_goal_idx_array = np.array([demo[intermediate_goal_idx[i]][0] * self.height, 
                                                                demo[intermediate_goal_idx[i]][1] * self.width]).astype(np.int32)
                        
                        self.goals.append(intermediate_goal_idx_array)
                    
                    goal_state_array = np.array([goal_state[0] * self.height, 
                                                 goal_state[1] * self.width]).astype(np.int32) 
                     
                    self.goals.append(goal_state_array)
        
        elif len(self.goals) != 0:
            if reset_goal:
                # If the number of goals is not 0 and we want to reset the goal,
                # we initialize an empty list of goals.             
                self.goals = []
                
                # If the goal_state expressed in the reset() function is None, 
                # we initialize a random position for the goal and ensure that the first 
                # position of the agent is not the same as the position 
                # of this goal. After this, we add this goal to the list of goals                     
                if goal_state is None:
                    goal = np.array([np.random.randint(0, self.height), 
                                     np.random.randint(0, self.width)]).astype(np.int32)
                    
                    while (self.state == goal).all():
                        goal = np.array([np.random.randint(0, self.height), 
                                         np.random.randint(0, self.width)]).astype(np.int32)
                        
                    self.goals.append(goal)
 
                elif goal_state is not None:
                    
                    # If the goal_state expressed in the reset() function is not None, 
                    # and if the number of goals is 1, we convert the normalized goal 
                    # back to it's original scale and append it to the list of goals.                     
                    if self.num_goals == 1:
                        goal_state[0] = goal_state[0] * self.height
                        goal_state[1] = goal_state[1] * self.width
                        self.goals.append(np.array(goal_state).astype(np.int32))
                    
                    # If the total number of the goals in the environment is more than 1
                    # calculate the positions of the intermediate goals in the demonstration, 
                    # convert the coordinates of the positions of the intermediate goals 
                    # back to the original scale and append them to a list of goals. 
                    
                    # In addiiton, if the number of goals is more than 1, 
                    # we treat the goal_state expressed as the parameter of the 
                    # reset() function as the final goal, convert the coordiantes 
                    # of it back to the original scale and append these coordiantes 
                    # to the list of goals. This will be the final goal of the agent.
                    elif self.num_goals > 1:
                        intermediate_goal_idx = []
                        for i in range(self.num_goals-1): 
                            goal_idx = (i + 1) * len(demo) // self.num_goals
                            intermediate_goal_idx.append(goal_idx)
                            
                            intermediate_goal_idx_array = np.array([demo[intermediate_goal_idx[i]][0] * self.height, 
                                                                    demo[intermediate_goal_idx[i]][1] * self.width]).astype(np.int32)
                            
                            self.goals.append(intermediate_goal_idx_array)
                        
                        goal_state_array = np.array([goal_state[0] * self.height, 
                                                     goal_state[1] * self.width]).astype(np.int32) 
                        
                        self.goals.append(goal_state_array)

        # Initialize a 2D grid where each point represents whether the corresponding 
        # location in the environment contains an obstacle, is assigned as a goal, 
        # or is safe to visit. 
        # In this observation grid:
        # 0 indicates that the location can be visited,
        # 1 indicates that the location contains an obstacle, and
        # 2+ indicates that the location is assigned as a goal.
        # (e.g., 2 represents the first goal, 3 represents the second goal, etc.)
        # At first, observation grid is initialized with 0s fully because we are resetting everything.
        self.observation = np.zeros((self.height, self.width)).astype(np.uint8)		
        
        # If the block variable is not None and assigned regions 
        # that indicate where to put an obstacle, 
        # we reflect those obstacles in the observation grid initialized above as 1.
        if self.block is not None:
            for region in self.block:
                block_hmin, block_hmax = int(region[0]), int(region[1])
                block_wmin, block_wmax = int(region[2]), int(region[3])
                for h in range(block_hmin, block_hmax+1):
                    for w in range(block_wmin, block_wmax+1):
                        self.observation[h, w] = 1
                        
        # For each goal in the goal list 
        # define a region that surrounds the location of that goal
        # and reflect the region that surrounds the location of the goal 
        # in the observation grid initialized above.
        for idx, goal in enumerate(self.goals):
            goal_hmin = int(goal[0] - 10)
            goal_hmax = int(goal[0] + 10)
            goal_wmin = int(goal[1] - 10)
            goal_wmax = int(goal[1] + 10)
            goal_hmin = max(0, goal_hmin)
            goal_hmax = min(self.height - 1, goal_hmax)
            goal_wmin = max(0, goal_wmin)
            goal_wmax = min(self.width - 1, goal_wmax)
            for h in range(goal_hmin, goal_hmax+1):
                for w in range(goal_wmin, goal_wmax+1):
                    self.observation[h,w] = 2 + idx
                    
        
        # Set the number of goals that are reached to 0 
        # because we are resetting everything in this function. 
        self.num_goals_reached = 0
        
        # Set the current step to 0.
        self._step = 0

        # Normalize the initial state.
        state = np.array([self.state[0] / self.height, 
                          self.state[1] / self.width]).astype(np.float32)
        
        # Return the initial state.
        return state


    def render(self, mode='', width=None, height=None):
        # Create a blank white image that is in the same shape as the environment.
        img = np.ones(self.observation.shape).astype(np.uint8) * 255
        
        # Identify the blocked region that has obstacles.
        blocked = np.where(self.observation == 1)
        
        # Color the obstacles with black.
        img[blocked] = 0


        # For each goal in the goa list,
        # calculate the boundaries of the goal location,
        # and color the goal area with dark gray.
        for idx, goal in enumerate(self.goals):
            hmin = max(0, goal[0] - 10)
            hmax = min(self.height - 1, goal[0] + 10)
            wmin = max(0, goal[1] - 10)
            wmax = min(self.width-1, goal[1] + 10)
            hmin = int(hmin)
            hmax = int(hmax)
            wmin = int(wmin)
            wmax = int(wmax)
            img[hmin:hmax, wmin:wmax] = 64

        # Color the region where agent is located in the environment with medium gray.
        img[max(0, int(self.state[0])-5): min(self.height-1, int(self.state[0])+5), 
            max(0, int(self.state[1])-5): min(self.width-1, int(self.state[1])+5)] = 128

        # If dimensions are specified to resize the image, 
        # resize the image by using those dimensions.
        if width is not None and height is not None:
            dim = (int(width), int(height))
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            
        # Add a third dimension to the image to represent the channel.
        img = img[..., None]

        # Return the image as array or display it depending on the mode variable.
        if mode=='rgb_array':
            return img
        else:
            cv2.imshow("Render", img)
            cv2.waitKey(5)

if __name__ == '__main__':
    # Create the environment with specified parameters.
    env = ParticleEnv(height = 640, 
                      width = 640, 
                      step_size = 10, 
                      reward_type = 'dense', 
                      reward_scale = None, 
                      start = None, 
                      goal = None, 
                      block = None)
    
    # Run 10 episodes and in each episode:
    for i in range(10):
        # Reset the environment.
        state = env.reset()
        
        # Explicitly show that the goal is not reached yet.
        done = False
        
        # And until the agent reaches the goal:
        while not done:
            # Take a random action. 
            action = env.action_space.sample()
            
            # Retrieve the next state, and reward 
            # after taking the action along with the information of 
            # whether the goal is reached and the action was successful.
            next_state, reward, done, info = env.step(action)
            
            # Display the current state of the environment.
            env.render()
            
            # Print the information about this step.
            print("State: ", state, 
                  "Action: ", action, 
                  "Next State: ", next_state, 
                  "Reward: ", reward, 
                  "Done: ", done, 
                  "Info: ", info)
            
            # Update the state
            state = next_state
        
        # After the agent reaches a goal, 
        # print the episode number along with the 
        # fina final state and reward. 
        print("Episode: ", i)
        print("Final State: ", state)
        print("Final Reward: ", reward)
        print("Final Done: ", done)
        print("Final Info: ", info)
        print("\n\n\n")