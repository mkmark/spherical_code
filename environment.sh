# alias conda="micromamba"

conda create -n spherical_code python=3.10 -c conda-forge -y
conda activate spherical_code

## for python batch solver
conda install pygments prompt_toolkit psutil packaging numpy tqdm matplotlib scipy -c conda-forge -y
## https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html
pip install -U 'celery[redis]'

## for python batch solver management
## https://parallel-ssh.org/
pip install parallel-ssh
## https://flower.readthedocs.io/en/latest/install.html#installation
pip install flower django pyjwt

## for gpu parallel
## https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
## https://github.com/taichi-dev/taichi
pip install taichi

## for result visulization
conda install pandas

conda env export > environment.yml
