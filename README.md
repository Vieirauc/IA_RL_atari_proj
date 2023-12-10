Exploring Deep Q Learning Dynamics in Atari Breakout:
A Comparative Analysis of Architectural Approaches

This guide provides step-by-step instructions on how to install the necessary systems, as well as the recreation of the experiments performed by our group.

**Step 1**:
- Guarantee that Python is installed on your machine, we can verify by running the command `pip --version` on a terminal. 
- In case it is not, utilize the following link to find the version of Python that is suited for your machine: https://www.python.org/downloads/

**Step 2**: 
- Install Cuda.
- Visit: https://developer.nvidia.com/cuda-12-1-0-download-archive
- Input your system settings and download the .exe file that is provided. 
- To verify that the installation was successful run the `nvcc --version` command.

**Step 3**:
- Install Pytorch. 
- Following the link: https://pytorch.org/ and scrolling down to the table that shows the installation guide, we can see the multiple parameters that PyTorch can be installed with, in our case (windows 11), the command that is generated is `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. We recommend that you use the same command to avoid mismatches in code versions. 
- To test if Pytorch was successfully installed, run Python and execute the `import torch` command. 
- As for Cuda support, run `torch.cuda.is_available()` as well as the `torch.cuda.get_device_name(0)` verifying that cuda has your GPU.

**Step 4**:
After the installation of the 
