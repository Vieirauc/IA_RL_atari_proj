# Exploring Deep Q Learning Dynamics in Atari Breakout: A Comparative Analysis of Architectural Approaches

## Installation

This guide provides step-by-step instructions on how to install the necessary systems, as well as the recreation of the experiments performed by our group.

**Step 1**:
- Guarantee that Python is installed on your machine, we can verify by running the command `pip --version` on a terminal. 
- In case it is not, utilize the following link to find the version of Python that is suited for your machine: [Python](https://www.python.org/downloads/).

**Step 2**: 
- Install Cuda. This improves the computing performance and aids the project.  
- Visit: [Cuda](https://developer.nvidia.com/cuda-12-1-0-download-archive).
- Input your system settings and download the .exe file that is provided. 
- To verify that the installation was successful run the `nvcc --version` command.

**Step 3**:
- Install Pytorch. 
- Following the link: [PyTorch](https://pytorch.org/) and scrolling down to the table that shows the installation guide, we can see the multiple parameters that PyTorch can be installed with, in our case (windows 11), the command that is generated is `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. We recommend that you use the same command to avoid mismatches in code versions. 
- To test if Pytorch was successfully installed, run Python and execute the `import torch` command. 
- As for Cuda support, run `torch.cuda.is_available()` as well as the `torch.cuda.get_device_name(0)` verifying that cuda has your GPU.

**Step 4**:
- Although optional, the creation of a virtual environment is advised.
- To do so run `python -m venv /path/to/new/virtual/environment`
- Access the venv by running the `activate.bat` found in the venv path.

**Step 5**:
- Once all the files are in a workspace, run `pip install -r requirements.txt` to make sure all the dependencies are correctly installed on your machine.

**Step 6**:
To visualize the agent learning to play the game, run the _test.py_ file with the following parameters:
agent = Agent(model=model,
              device=device,
              epsilon=0.05,
              nb_warmup=1000,
              nb_actions=4,
              learning_rate=0.01,
              memory_capacity=1000000,
              batch_size=64)

To test the different influence that the hyperparameters have on the agents learning, you can alter the epsilon(maintain the values between 0.01 and 0.1), nb_warmup(can vary between 1000 and 4000) and the learning_rate(vary between 0.01 and 0.0001).


## Training

