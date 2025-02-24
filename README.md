# rl-models
Collection of reinforcement learning models made with PyTorch.

These RL models are trained to solve environments from [Gymnasium]([url](https://gymnasium.farama.org/)), which provides standardized environments.

There are currently 5 models:
1. Double dueling deep Q network trained on lunar lander v-3 from Gymansium
2. Double dueling deep convolutional Q network for pac man from Gymnasium
3. A2C model for kung fu master from Gymnasium
4. PPO model for car racing from Gymnasium
5. SAC model for pendulum from Gymnasium

The first 2 models both use gradient clipping to stabilize the model during training and prevent them from being stuck in a local mimimum.

The Pac-man model also uses batch normalization to speed up training and improve accuracy.

The third model:
- dynamically computes the feature size
- uses preprocessing to combine 4 frames into a grayscale stack
- uses dynamic rewards normalization based on a moving average to stabilize training

The 4th model uses a PPO implementation from [stable-baselines3]([url](https://github.com/DLR-RM/stable-baselines3)), which contains PyTorch implementations of various RL algorithms

The 5th model uses a SAC implementation from [stable-baselines3]([url](https://github.com/DLR-RM/stable-baselines3)).

## Usage Instructions
```
git clone https://github.com/avanishd-3/rl-models.git
```

Then run the notebooks

## Lunar lander
Running the notebook will train the AI and run it on a test run of lunar landing.

There is also a saved model weight and a video showing that model's performance on a test run, which I did.


## Pac Man 
Running the notebook will train the AI and run it on a test run of pac man.

This model does not have a saved model weight and video, because I don't have the specs (CUDA) to run this model quickly.

## Kung Fu Master
Running the notebook will train the AI and run it on a test run of kung fu master.

This model does not have a saved model weight, because it is pretty cheap and quick to implement (even on only CPU).

It does have a video showing the model's performance on a test run I did (the model got a score of 2400).

## Car Racing
This is a pretty complex (i.e. compute intensive) environment, since inputs are continuous. So, CUDA is definitely a requirement when running this notebook.

There is a saved model weight in the zip file based on 50,000 time steps of training (model performance will increase if more timesteps are used).

This model can be loaded by using

```
model = PPO.load(ppo_car_racing)
```

There is also a video showing the model's performance on a test run I did.

## Pendulum
There is a saved model weight in the zip file based on 20,000 time steps of training (this is enough for pendulum).

This model can be loaded by using

```
model = PPO.load(sac_pendulum)
```

There is also a video showing the model's performance on a test run I did.

## References

Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
  Learning with a Stochastic Actor. arXiv preprint arXiv:1801.01290.

Hasselt, H.V., Guez, A., Silver, D. (2015). Deep Reinforcement Learning with Double Q-learning. arXiv preprint arXiv:1509.06461.

Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T.P., Harley, T., Silver, D., Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. arXiv preprint arXiv:1602.01783.

Ioffe, S., Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv preprint arXiv:1502.03167.

O'Shea, K., Nash, R. (2015). An Introduction to Convolutional Neural Networks. arXiv preprint arXiv:1511.08458.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.

Wang, Z., Schaul, T., Hessel, M., Hasselt, H.V., Lanctot, M., Freitas, N.D. (2015). Dueling Network Architectures for Deep Reinforcement Learning. arXiv preprint arXiv:1511.06581.

Zhang, J., He, T., Sra, S., Jadbabaie, A. (2019). Why gradient clipping accelerates training: A theoretical justification for adaptivity. arXiv preprint arXiv:1905.11881.


