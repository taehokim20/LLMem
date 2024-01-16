# LLMem: GPU Memory Estimation for Fine-Tuning Pre-Trained LLMs

Our source code is based on an efficient AI-model training system (https://github.com/hpcaitech/ColossalAI). \
The models used are [facebook/opt](https://huggingface.co/docs/transformers/model_doc/opt), [bigscience/bloom](https://huggingface.co/docs/transformers/model_doc/bloom), [Salesforce/codegen](https://huggingface.co/docs/transformers/model_doc/codegen), [microsoft/biogpt](https://huggingface.co/docs/transformers/model_doc/biogpt), [bigcode/gpt_bigcode-santacoder](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode), and [EleutherAI/gpt-neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo). \
The dataset used is alpaca data (https://github.com/tatsu-lab/stanford_alpaca), which is 52K instruction-following data.

# Abstract
Fine-tuning pre-trained LLMs with limited hardware faces memory constraints. Several distributed fine-tuning methods have been proposed to alleviate memory constraints. However, we do not know which method is the best for fast fine-tuning while avoiding out-of-memory in a given environment. We propose LLMem, which estimates the memory consumption when applying distributed fine-tuning methods to multiple GPUs and informs the optimal method and batch size. We complete the memory estimation before fine-tuning based on the basic structure of transformer-based decoder models and the memory usage distribution of each method. The experimental results show that LLMem estimates peak GPU memory on a single GPU with error rates of up to 1.6\%. In addition, when applying distributed fine-tuning methods to LLMs with more than a billion parameters on multiple GPUs, it shows an average error rate of 3.0\%.

# How to set up
## cuda-11.7
1. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
2. sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
3. wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
4. sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
5. sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
6. sudo apt-get update
7. sudo apt-get -y install cuda
8. In ~/.bashrc, \
      export CUDA_HOME=/usr/local/cuda \
      export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH \
      export PATH=$PATH:$CUDA_HOME/bin
9. source ~/.bashrc
10. (Check if it works) nvidia-smi, nvcc --version

## Anaconda3
1. sudo apt-get update
2. wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
3. sha256sum Anaconda3-2023.03-Linux-x86_64.sh
4. bash Anaconda3-2023.03-Linux-x86_64.sh
5. source ~/.bashrc
6. conda create --name colo201 python==3.10.0
7. conda activate colo201

## NCCL 2.14.3 for CUDA 11.7
1. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
2. sudo dpkg -i cuda-keyring_1.0-1_all.deb
3. sudo apt-get update

## Colossal-AI
1. git clone https://github.com/hpcaitech/ColossalAI.git
2. cd ColossalAI/
3. git reset --hard d4fb7bfda7a2da5480e1187e8d3e40884b42ba11
4. cp -r ../LLMem/colossalai .
5. pip install .

## Others
1. pip install transformers==4.29.2
2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
3. pip install GPUtil pynvml sentencepiece datasets deepspeed sacremoses openai==0.27.9
4. pip install --upgrade accelerate

## How to run
1. In the LLMem directory, move the stuff to the ColossalAI directory \
         cp *.py run_colo.sh alpaca_data.json ../ColossalAI/
3. cd ../ColossalAI
### Measure ground truth
1. Change the model from ~/anaconda3/envs/colo201/lib/python3.10/site-packages/transformers/models/xxx/yyy.py to ./real_models/yyy.py
2. For DP, run bash run_colo.sh after changing the number of nodes, model name, per_device_train_batch_size with dp_real.py
3. For TP or DP+TP, tp_real.py also follows the similar process. but it requires to change tp_size in tp_real.py \
   For example, the number of nodes = 2 and tp_size = 2 -> 2DP, the number of nodes = 4 and tp_size = 2 -> 2DP+2TP. \

*For the ground truth, you should add (the measured GPU memory by nvml - the memory by GPUtil) to the peak GPU memory.
### Estimate peak GPU memory
1. Use the original model, not including the GPU memory measurement part
2. Set up the values in run_colo.sh and tp_size in tp_real.py (if you are applying tensor parallelism)
3. For DP, uncomment lines 340-353 and lines 359-367 in dp_real.py and run bash run_colo.sh
4. For DP, uncomment lines 379-392 and lines 398-407 in tp_real.py and run bash run_colo.sh
