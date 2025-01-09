# rl-models
Collection of reinforcement learning models made with PyTorch.

These RL models are trained to solve environments from [Gymnasium]([url](https://gymnasium.farama.org/)), which provides standardized environments.

There are currently 2 models: a double dueling deep Q network trained on lunar lander v-3 from Gymansium, and a double dueling deep convolutional Q network for pac man from Gymnasium.

Both models use gradient clipping to stabilize the model during training and prevent it from being stuck in a local mimimum.

The Pac-man model also uses batch normalization to speed up training and improve accuracy.

## Usage Instructions
1. git clone https://github.com/avanishd-3/rl-models.git
2. Run the notebooks

## Lunar lander
Running the notebook will train the AI and run it on a test run of lunar landing.

There is also a saved model weight and a video showing that model's performance on a test run, which I did.


## Pac Man 
Running the notebook will train the AI and run it on a test run of lunar landing.

This model does not have a saved model weight and video, because I don't have the specs (CUDA) to run this model quickly.

## References

Hasselt, H.V., Guez, A., Silver, D. (2015). Deep Reinforcement Learning with Double Q-learning. arXiv preprint arXiv:1509.06461.

Ioffe, S., Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv preprint arXiv:1502.03167.

O'Shea, K., Nash, R. (2015). An Introduction to Convolutional Neural Networks. arXiv preprint arXiv:1511.08458.

Wang, Z., Schaul, T., Hessel, M., Hasselt, H.V., Lanctot, M., Freitas, N.D. (2015). Dueling Network Architectures for Deep Reinforcement Learning. arXiv preprint arXiv:1511.06581.

Zhang, J., He, T., Sra, S., Jadbabaie, A. (2019). Why gradient clipping accelerates training: A theoretical justification for adaptivity. arXiv preprint arXiv:1905.11881.


