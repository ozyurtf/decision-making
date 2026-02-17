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

![Behavior Cloning](figures/bc.png)

<div align="center">
  <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start;">
    <div style="text-align: center; max-width: 245px;">
      <h4>Fixed Goal</h4>
      <img src="./gifs/behavior-cloning/fixed/fixed.gif" alt="Fixed Goal" width="245" style="border: 3px solid #666666;"/>
      <p style="font-size: 14px; text-align: left;">In the fixed goal dataset, the same observation always maps to the same action and the MLP can learn this mapping easily because there is no conflicting signal. That's why it performs well on the fixed goal dataset.</p>
    </div>
    <div style="text-align: center; max-width: 245px;">
      <h4>Changing Goal</h4>
      <img src="./gifs/behavior-cloning/changing/changing.gif" alt="Changing Goal" width="245" style="border: 3px solid #666666;"/>
      <p style="font-size: 14px; text-align: left;">In the changing goal dataset, the same observation can map to different actions in different episodes. That's why the MLP fails to learn this mapping and performs poorly on the changing goal dataset.</p>
    </div>
    <div style="text-align: center; max-width: 245px;">
      <h4>Multimodal</h4>
      <img src="./gifs/behavior-cloning/multimodal/multimodal.gif" alt="Multimodal" width="245" style="border: 3px solid #666666;"/>
      <p style="font-size: 14px; text-align: left;">In the multimodal dataset, the last goal is fixed but the agent should reach other goals first before reaching the last goal. Therefore, like in the changing goal dataset, the same observation can map to different actions (because of different intermediate goals) in different episodes. That's why the MLP fails to learn this mapping and performs poorly on the multimodal dataset.</p>
    </div>
  </div>
</div>

## Goal Conditioned Behavior Cloning

![Goal-Conditioned Behavior Cloning](figures/gcbc.png)

<div align="center">
  <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start;">
    <div style="text-align: center; max-width: 245px;">
      <h4>Fixed Goal</h4>
      <img src="./gifs/goal-conditioned-behavior-cloning/fixed/fixed.gif" alt="Fixed Goal" width="245" style="border: 3px solid #666666;"/>
      <p style="font-size: 14px; text-align: left;">In the fixed goal dataset, the model still performs well for the same reasons as in behavior cloning.</p>
    </div>
    <div style="text-align: center; max-width: 245px;">
      <h4>Changing Goal</h4>
      <img src="./gifs/goal-conditioned-behavior-cloning/changing/changing.gif" alt="Changing Goal" width="245" style="border: 3px solid #666666;"/>
      <p style="font-size: 14px; text-align: left;">In the changing goal dataset, the model performs well because when the observation is the same and the action is different in different episodes, the goal is different as well and this makes the input to the model unique. Therefore, the MLP can learn the mapping from observation and goal to action in a goal-conditioned manner.</p>
    </div>
    <div style="text-align: center; max-width: 245px;">
      <h4>Multimodal</h4>
      <img src="./gifs/goal-conditioned-behavior-cloning/multimodal/multimodal.gif" alt="Multimodal" width="245" style="border: 3px solid #666666;"/>
      <p style="font-size: 14px; text-align: left;">In the multimodal setting, even though intermediate goals can be different in different episodes, since we are feeding the next goal as input to the model at each step, the model performs well on this dataset as well.</p>
    </div>
  </div>
</div>

## Behavior Transformer

![Behavior Transformer](figures/bet.png)

<div align="center">
  <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start;">
    <div style="text-align: center; max-width: 245px;">
      <h4>Fixed Goal</h4>
      <img src="./gifs/behavior-transformer/fixed/fixed.gif" alt="Fixed Goal" width="245" style="border: 3px solid #666666;"/>
      <p style="font-size: 14px; text-align: left;">In the fixed goal dataset, since the goal is the same across all episodes, for any given observation [x, y], there's only one correct action direction which is toward that fixed goal. This means that the expert actions for the same observation always belong to the same cluster. The ground truth label is always cluster i and it never conflicts with cluster j. The binning head learns this consistent mapping easily, argmax picks the right cluster every time, the offset refines it to the precise action, and the agent learns to reach the goal easily.</p>
    </div>
    <div style="text-align: center; max-width: 245px;">
      <h4>Changing Goal</h4>
      <img src="./gifs/behavior-transformer/changing/changing.gif" alt="Changing Goal" width="245" style="border: 3px solid #666666;"/>
      <p style="font-size: 14px; text-align: left;">In the changing goal setting, we see that the model performs very poorly because of similar reasons as in behavior cloning. Let's say the agent is at [x, y] position in episode a and episode b of the changing goal setting. Since the goals will be different, the expert actions will be different as well and therefore they will be associated with different clusters. Let's call them cluster i and cluster j. In other words, the ground truth label for the same observation might be either cluster i or cluster j. Let's say that cluster i represents "go toward top left", cluster j represents "go toward right". When the agent is at [x, y], and argmax picks cluster j that puts the agent in the wrong direction, the agent will never reach the target. And this is what we see in the gif above.</p>
    </div>
    <div style="text-align: center; max-width: 245px;">
      <h4>Multimodal</h4>
      <img src="./gifs/behavior-transformer/multimodal/multimodal.gif" alt="Multimodal" width="245" style="border: 3px solid #666666;"/>
      <p style="font-size: 14px; text-align: left;">The same things can be said in the multimodal setting as well. In the multimodal setting, there are intermediate goals that change in different episodes and one final goal that is the same in each episode. Similar to the changing goal setting, the agent picks a cluster with argmax and that cluster can sometimes put the agent in the wrong direction and cause it to miss the intermediate goals. In the gif above, we see that the agent always follows the same path regardless of the positions of the intermediate goal, which is correct for some episodes and wrong for others. The problem here is at inference. The model needs to pick the right cluster for the current episode but it doesn't have that information (the intermediate goal) to make that choice. And since the final goal is always fixed, the agent is still able to reach the final goal even when it misses the right intermediate goals because every cluster that was learned from the expert data ultimately leads the agent to the final goal.</p>
    </div>
  </div>
</div>

## References

- Nur Muhammad Mahi Shafiullah, Zichen Jeff Cui, Ariuntuya Altanzaya, Lerrel Pinto. "Behavior Transformers: Cloning *k* modes with one stone." *Thirty-Sixth Conference on Neural Information Processing Systems*, 2022. [[Paper]](https://openreview.net/forum?id=agTr-vRQsa)
