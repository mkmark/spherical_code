# %%

ver = "v1_0_0"

ns = range(4, 20)
"""the range of n to calculate within current run"""

min_min_count = 10
"""
this value determines the minimum occurence of a minimum solution
before it is deemed as a global minimum
"""

out_base_path = f"out/{ver}/min_potential"

sqlite3_db_path = f"{out_base_path}/db.sqlite3"

solver_path = f"src/{ver}/solver/solver_min_potential.o3.out"

txt_dump_path = f"{out_base_path}/txt_dump/"

# %% init
import os
import sqlite3
if not os.path.exists(sqlite3_db_path):
  con = sqlite3.connect(sqlite3_db_path)
  cur = con.cursor()
  cur.execute("CREATE TABLE result(n INT, seed INT, value REAL, step INT, diff REAL)")
  cur.execute("CREATE UNIQUE INDEX idx_n_seed ON result (n, seed);")
  cur.execute("CREATE INDEX idx_n ON result (n);")
  con.close()

con = sqlite3.connect(sqlite3_db_path)
cur = con.cursor()

# %%
def get_dict_with_default(d, n, default=0):
  if n in d:
    return d[n]
  else:
    d[n] = default
    return default

def get_status():
  total_cnts = {}
  min_cnts = {}
  min_vals = {}
  min_val_seeds = {}

  cur.execute("SELECT * FROM result")
  records = cur.fetchall()

  for record in records:
    n, seed, value, step, diff = record
    total_cnts[n] = get_dict_with_default(total_cnts, n) + 1
    rounded_value = round(value, 7)
    min_vals_n = get_dict_with_default(min_vals, n, float("inf"))
    if rounded_value == min_vals_n:
      min_cnts[n] = get_dict_with_default(min_cnts, n) + 1
    elif rounded_value < min_vals_n:
      min_vals[n] = rounded_value
      min_cnts[n] = 1
      min_val_seeds[n] = seed
  
  return total_cnts, min_cnts, min_vals, min_val_seeds


import numpy as np
def get_batch_size(n):
  """
  `batch_size` is the estimated minimun random initial trial count
  to ensure a 99% confidence to get at least one global minimum.
  This is a function estimated
  utlizing the fact that given a large enough number of random initial trials,
  one will get an accurate probability $P(n)$ of getting a global minimum out of all trials.
  Assuming there exists a continuous lower bound of $P$, ensuring $p(n) <= P(n)$,
  the batch_size $b$ is then predicted per $(1-p(n))^b < 1-0.99$, i.e. $b > log_{1-p(n)}{0.01}$.
  Thus, this function is reviewed and updated manually as $n$ gets larger.
  However, one can never be fully confident that the number is large enough,
  so the result is still an estimate at best.
  It is estimated that $p(n)=1/(1 + 1.85404472e-05*n**4)$ according to $n < 256$.
  """
  # return n
  return int(-2/np.log10(1 - 1/(1 + 1.85404472e-05*n**4))) + 1

# %%
import tqdm
import multiprocessing as mp

import subprocess
import os

def exec_cmd(cmd):
  return subprocess.check_output(cmd.split(" "), preexec_fn=lambda : os.nice(19)).decode()

# %%
total_cnts, min_cnts, min_vals, min_val_seeds = get_status()

# %%
is_to_continue = True
while is_to_continue:
  is_to_continue = False
  tasks = []
  for n in ns:
    batch_size = get_batch_size(n)
    if get_dict_with_default(min_cnts, n) < min_min_count \
      or get_dict_with_default(total_cnts, n) < batch_size:
        is_to_continue = True
        initial_seed = get_dict_with_default(total_cnts, n)
        target_total_cnt = (initial_seed//batch_size + 1) * batch_size
        for i in range(initial_seed, target_total_cnt):
          tasks.append((n, i, ""))
        total_cnts[n] += target_total_cnt

  cmds = [f"{solver_path} {task[0]} {task[1]}" for task in tasks]
  with mp.Pool() as pool:
    res = list(tqdm.tqdm(pool.imap(exec_cmd, cmds), total=len(tasks)))

  for i, task in enumerate(tasks):
    n, seed, _ = task
    value, step, diff, _ = res[i].split("\n")
    cur.execute(f"""
      REPLACE INTO result VALUES
        ({n}, {seed}, {value}, {step}, {diff})
    """)
    
  total_cnts, min_cnts, min_vals, min_val_seeds = get_status()

tasks = []
for n in ns:
  tasks.append((n, min_val_seeds[n], txt_dump_path))

cmds = [f"{solver_path} {task[0]} {task[1]} {task[2]}" for task in tasks]
with mp.Pool() as pool:
  pool.map(exec_cmd, cmds)

# %%
cur.close()
con.commit()
con.close()


# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
x = list(total_cnts.keys())
y = np.array(list(total_cnts.values()))/np.array(list(min_cnts.values()))
plt.figure()
plt.bar(
  x,
  y,
)
# %%
x_max = [4, 37, 78, 178, 248]
y_max = [1, 6, 117, 17800, 71548]
f = interp1d(x_max, y_max, kind='quadratic')
x_smooth = np.linspace(4, 248, 1000)
y_smooth = f(x_smooth)
plt.plot(x, y, 'ro', label='Data')
plt.plot(x_smooth, y_smooth, 'b-', label='Interpolated Curve')

# %%
from scipy.optimize import curve_fit
def polynomial_func(x, a, b):
    return a + b * x**4
params, _ = curve_fit(polynomial_func, x_max, y_max, sigma=np.square(x_max))
x_fit = np.linspace(4, 255, 1000)
y_fit = polynomial_func(x_fit, *params)
plt.plot(x, y, 'ro', label='Data')
plt.plot(x_fit, y_fit, 'b-', label='Interpolated Curve')

# %%
