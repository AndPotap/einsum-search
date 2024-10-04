# GPT experiments
conda create -y -n gpt python=3.10 --force
conda activate gpt
pip install torch --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
pip install numpy transformers datasets tiktoken wandb tqdm torchmetrics --no-cache-dir
cd $HOME
git clone https://github.com/wilson-labs/cola.git
cd -
pip install $HOME/cola/.[dev]
pip install -U fvcore torchinfo einops opt_einsum
conda deactivate

# CIFAR experiments
# We need to install FFCV, which is pretty painful and it's going to take a while
conda env create -y -f conda_env.yml # env name will be "struct"
conda activate struct
pip install --upgrade pip
pip uninstall torch -y
pip uninstall torchvision -y
pip install torch==1.12.0 torchvision==0.13.0 torchdistx --index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
# conda install -y -c pytorch -c conda-forge torchdistx cudatoolkit=11.6 -f
# conda install -y -c pytorch -c conda-forge torchdistx cudatoolkit=11.6 --force-reinstall
cd ~; git clone https://github.com/wilson-labs/cola.git; cd -
sed -i '/torch\.func/s/^/#/' ~/cola/cola/backends/torch_fns.py # comment out lines containing torch.func which is not available in torch 1.12
pip install $HOME/cola/.[dev]
pip install datasets tiktoken torchmetrics
conda deactivate

