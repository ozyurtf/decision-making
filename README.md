**Note**: I took the core codebase for environment simulation and related infrastructure in this repository from my assignments in the Deep Decision Making and Reinforcement Learning (DDRL) course at New York University. I only implemented the Behavior Cloning, Goal-Conditioned Behavior Cloning, DAgger, Double Q-Learning, Dueling DQN, and PPO algorithms. Their performance are shown in the GIFs below. 

## Environment 

<div align="center">
  <img src="figures/environment.png" alt="Environment" width="400"/>
</div>

## Expert Dataset 

### Changing Goal 

**Episode 1**

![Changing Goal, Episode 1](figures/changing_goal/episode_0.png)

**Episode 2**

![Changing Goal, Episode 2](figures/changing_goal/episode_1.png)

**Episode 3**

![Changing Goal, Episode 3](figures/changing_goal/episode_2.png)

### Fixed Goal 

**Episode 1**

![Fixed Goal, Episode 1](figures/fixed_goal/episode_0.png)

**Episode 2**

![Fixed Goal, Episode 1](figures/fixed_goal/episode_1.png)

**Episode 3**

![Fixed Goal, Episode 2](figures/fixed_goal/episode_2.png)

### Multimodal

**Episode 1**

![Multimodal, Episode 1](figures/multimodal/episode_0.png)

**Episode 2**

![Multimodal, Episode 2](figures/multimodal/episode_1.png)

**Episode 3**

![Multimodal, Episode 3](figures/multimodal/episode_2.png)

## Behavior Cloning

![Behavior Cloning](figures/behavior-cloning.png)

<div align="center">
  <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start;">
    <div style="text-align: center;">
      <h4>Fixed Goal</h4>
      <img src="./gifs/behavior-cloning/fixed/fixed.gif" alt="Fixed Goal" width="245" style="border: 3px solid #666666;"/>
    </div>
    <div style="text-align: center;">
      <h4>Changing Goal</h4>
      <img src="./gifs/behavior-cloning/changing/changing.gif" alt="Changing Goal" width="245" style="border: 3px solid #666666;"/>
    </div>
    <div style="text-align: center;">
      <h4>Multimodal</h4>
      <img src="./gifs/behavior-cloning/multimodal/multimodal.gif" alt="Multimodal" width="245" style="border: 3px solid #666666;"/>
    </div>
  </div>
</div>

In the fixed goal dataset, the same observation always maps to the same action and MLP can learn this mapping easily because there is no conflicting signal. That's why it performs well on fixed goal dataset. 

In the changing goal dataset, the same observation can map to different actions in different episodes. That's why MLP fails to learn this mapping and performs poorly on changing goal dataset.

In the multimodal dataset, the goal is fixed but there are different paths to reach the goal. Therefore, like in changing goal dataset, the same observation can map to different actions in different episodes. That's why MLP fails to learn this mapping and performs poorly on multimodal dataset as well.

## Goal Conditioned Behavior Cloning

![Goal-Conditioned Behavior Cloning](figures/goal-conditioned-behavior-cloning.png)

<div align="center">
  <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start;">
    <div style="text-align: center;">
      <h4>Fixed Goal</h4>
      <img src="./gifs/goal-conditioned-behavior-cloning/fixed/fixed.gif" alt="Fixed Goal" width="245" style="border: 3px solid #666666;"/>
    </div>
    <div style="text-align: center;">
      <h4>Changing Goal</h4>
      <img src="./gifs/goal-conditioned-behavior-cloning/changing/changing.gif" alt="Changing Goal" width="245" style="border: 3px solid #666666;"/>
    </div>
    <div style="text-align: center;">
      <h4>Multimodal</h4>
      <img src="./gifs/goal-conditioned-behavior-cloning/multimodal/multimodal.gif" alt="Multimodal" width="245" style="border: 3px solid #666666;"/>
    </div>
  </div>
</div>

In the goal conditioned behavior cloning, we use observation and goal as input to the MLP and try to learn the mapping from observation and goal to action. This way, the MLP can learn the mapping from observation to action in a goal-conditioned manner. 

In the fixed dataset, the model still performs well because of the same reasons as in behavior cloning. 

In the changing dataset, the model performs well because when the observation is the same and the action is different in different episodes, the goal is different as well and this makes the input to the model unique. Therefore, MLP can learn the mapping from observation and goal to action in a goal-conditioned manner.

But in the multimodal setting, the expert has multiple valid paths to the **same goal**. So, the model can see [x, y, $g_x$, $g_y$] with action "go left" in some episodes and [x, y, $g_x$, $g_y$] with action "go right" in other episodes even though the goal is the same. The model is back to the same problem as behavior cloning: the loss averages the conflicting actions, and µ lands somewhere between the two valid directions.

## Behavior Transformer

![Behavior Transformer](figures/bet.png)

<div align="center">
  <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start;">
    <div style="text-align: center;">
      <h4>Fixed Goal</h4>
      <img src="./gifs/behavior-transformer/fixed/fixed.gif" alt="Fixed Goal" width="245" style="border: 3px solid #666666;"/>
    </div>
    <div style="text-align: center;">
      <h4>Changing Goal</h4>
      <img src="./gifs/behavior-transformer/changing/changing.gif" alt="Changing Goal" width="245" style="border: 3px solid #666666;"/>
    </div>
    <div style="text-align: center;">
      <h4>Multimodal</h4>
      <img src="./gifs/behavior-transformer/multimodal/multimodal.gif" alt="Multimodal" width="245" style="border: 3px solid #666666;"/>
    </div>
  </div>
</div>
