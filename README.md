# Exploring Deep Q Learning Dynamics in Atari Breakout: A Comparative Analysis of Architectural Approaches

## Installation

This guide provides step-by-step instructions on how to install the necessary systems, as well as the recreation of the experiments performed by our group.

**Step 1**:
- Guarantee that Python is installed on your machine, we can verify by running the command `pip --version` on a terminal. 
- In case it is not, utilize the following link to find the version of Python that is suited for your machine: [Python](https://www.python.org/downloads/).
- We used Python 3.11.5 for our project

**Step 2**:
- Install Pytorch. 
- Following the link: [PyTorch](https://pytorch.org/) and scrolling down to the table that shows the installation guide, we can see the multiple parameters that PyTorch can be installed with, in our case (windows 11), the command that is generated is `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. We recommend that you use the same command to avoid mismatches in code versions.
- For our project, we utilized PyTorch version 2.1.1 alongside Cuda version 12.1.
- To test if Pytorch was successfully installed, run Python and execute the `import torch` command. 


**Step 3**: 
- Install Cuda. This improves the computing performance and aids the project.
- Ensure installation of the appropriate version of Cuda that is compatible with PyTorch
- Visit: [Cuda](https://developer.nvidia.com/cuda-12-1-0-download-archive).
- Input your system settings and download the .exe file that is provided. 
- To verify that the installation was successful run the `nvcc --version` command.
- As for Cuda support with torch, run `torch.cuda.is_available()` ,after importing torch, and run as well `torch.cuda.get_device_name(0)` verifying that your GPU is being detected as a Cuda device by torch.

**Step 4**:
- Although optional, the creation of a virtual environment is advised.
- To do so run `python -m venv /path/to/new/virtual/environment`
- Activate the venv by running the `activate.bat` found in the venv path.

**Step 5**:
- Once all the files are in a workspace, run `pip install -r requirements.txt` to make sure all the dependencies are correctly installed on your virtual environment or machine.

**Step 6**:
To visualize an agent playing the game, go to _test.py_ , write the path to the trained model in vairable _input_file_ and then  run the _test.py_ file with the following parameters:

agent = Agent(model=model,
              device=device,
              epsilon=0.05,
              nb_warmup=1000,
              nb_actions=4,
              memory_capacity=1000000,
              batch_size=64)

To test the different influence that the hyperparameters have on the agents learning, you can alter the epsilon(maintain the values between 0.01 and 0.1), and nb_warmup(can vary between 1000 and 4000).

Note: We recommend to test out best model that can be found here: models\pastmodels\simple_lr0001_mc100000_warm5000\model_iter_8000.pt


## Training
**Note**: All the input data is generated within our project's environment.

For the training section of the project, we can find the _main.py_ file that holds the project's core. From there we can load a model from the _model.py_ file where the neural network architecture, AtariNet and Pytorch nn module are defines. If there is a _latest.pt_ file available to load, the agent will continue the project from that point, otherwise it generates a new one and saves the information that is generated along the run. The plots are generated on _plot.py_ file while the agent is learning, providing the user with a visualization of the information that is being retrieved(epsilon and epochs).

The hyperparameters that we used to train the agent are ther following:

agent = Agent(model=model,
              device=device,
              epsilon=1.0,
              min_epsilon=0.1,
              nb_warmup=5000,
              nb_actions=4,
              learning_rate=0.01,
              memory_capacity=100000,
              batch_size=64)
              
## References
- [Gymnasium](https://gymnasium.farama.org/environments/atari/)
- [Dueling deep Q Network](https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-4-double-dqn-and-dueling-dqn-b349c9a61ea1)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Double DQN and Dueling DQN](https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-4-double-dqn-and-dueling-dqn-b349c9a61ea1)
- [Breakout_1](https://en.wikipedia.org/wiki/Breakout\_(video\_game))
- [Breakout 2](https://ultimatepopculture.fandom.com/wiki/Breakout\_(video\_game))
- [Pytorch 1](https://www.nvidia.com/en-us/glossary/data-science/pytorch/)
- [Pytorch 2](https://github.com/pytorch/pytorch)
- [Human level control DRL](https://doi.org/10.1038/nature14236)
- [Distributed prioritized](https://arxiv.org/abs/1803.00933)
