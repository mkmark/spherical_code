alias mm="micromamba"

mm create -n spherical_code python=3.10 -c conda-forge -y
mm activate spherical_code

mm install pygments psutil packaging numpy tqdm matplotlib scipy -c conda-forge -y
## https://pytorch.org/get-started/locally/
mm install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
## https://parallel-ssh.org/
pip install parallel-ssh
## https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html
pip install -U 'celery[redis]'
## https://github.com/taichi-dev/taichi
pip install taichi

mm env export > environment.yml
