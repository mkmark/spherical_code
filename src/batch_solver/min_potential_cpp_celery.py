# %%

ver = "v1.0.2"

ns = range(4, 300)
"""the range of n to calculate within current run"""

min_min_count = 20
"""
this value determines the minimum occurence of a minimum solution
before it is deemed as a global minimum
"""

max_task_queue_length = 32768

out_base_path = f"out/{ver}/min_potential"

sqlite3_db_path = f"{out_base_path}/db.sqlite3"

solver_path = f"release/{ver}/solver/SolverMinPotentialNaive.o3.out"

txt_dump_path = f"{out_base_path}/txt_dump/"

# %% init
import os
os.makedirs(out_base_path, exist_ok=True)
os.makedirs(txt_dump_path, exist_ok=True)

import sqlite3
if not os.path.exists(sqlite3_db_path):
  con = sqlite3.connect(sqlite3_db_path)
  cur = con.cursor()
  cur.execute("CREATE TABLE record(n INT, seed INT, value REAL, step INT, diff REAL);")
  cur.execute("CREATE UNIQUE INDEX idx_n_seed ON record (n, seed);")
  cur.execute("CREATE INDEX idx_n ON record (n);")
  cur.execute("CREATE INDEX idx_value ON record (value);")
  cur.execute("CREATE INDEX idx_seed ON record (seed);")
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

import tqdm

def get_status(
  ns=None,
  gen_missed_tasks=False
):
  print("get_status")

  total_cnts = {}
  min_cnts = {}
  min_vals = {}
  min_val_seeds = {}

  missed_tasks = []

  if not ns:
    ns = [n[0] for n in cur.execute("SELECT DISTINCT n FROM record;").fetchall()]
  for n in tqdm.tqdm(ns):
    total_cnt_n = cur.execute(f"SELECT COUNT(n) FROM record WHERE n={n};").fetchone()[0]
    total_cnts[n] = total_cnt_n
    if total_cnt_n == 0:
      continue
    min_val_n = cur.execute(f"SELECT MIN(value) FROM record WHERE n={n};").fetchone()[0]
    rounded_min_val_n = round(min_val_n, 7)
    min_vals[n] = rounded_min_val_n
    min_cnts[n] = cur.execute(f"SELECT COUNT(n) FROM record WHERE n={n} AND value<{min_val_n+1e-7};").fetchone()[0]
    # min_cnts[n] = cur.execute(f"SELECT COUNT(n) FROM record WHERE n={n} AND ROUND(value, 7)={rounded_min_val_n};").fetchone()[0]
    min_val_seeds[n] = cur.execute(f"SELECT seed FROM record WHERE n={n} AND value={min_val_n};").fetchone()[0]
    if gen_missed_tasks:
      max_seed_n = cur.execute(f"SELECT MAX(seed) FROM record WHERE n={n}").fetchone()[0]
      if max_seed_n != total_cnt_n-1:
        missed_tasks_cnt_n = max_seed_n + 1 - total_cnt_n
        print(f"missing record of n={n}: {missed_tasks_cnt_n}")
        seed = total_cnt_n-1
        pbar = tqdm.tqdm(total=missed_tasks_cnt_n)
        while missed_tasks_cnt_n != 0:
          if cur.execute(f"SELECT 1 FROM record WHERE n={n} AND seed={seed};").fetchone() is None:
            missed_tasks_cnt_n -= 1
            missed_tasks.append((n, seed, ""))
            pbar.update(1)
          seed -= 1
  
  return total_cnts, min_cnts, min_vals, min_val_seeds, missed_tasks

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
  return max(10000, int(-2/np.log10(1 - 1/(1 + 1.85404472e-05*n**4))/10) + 1)
  # return int(-2/np.log10(1 - 1/(1 + 1.85404472e-05*n**4))/10) + 1

  ## until v1.0.1
  # return int(-2/np.log10(1 - 1/(1 + 1.85404472e-05*n**4))) + 1

  # return n

# %%
from src.batch_solver.celery.tasks import exec_cmd

import celery
from celery import Celery, group
from celery.app.task import Task
from celery.result import GroupResult, AsyncResult
from celery.exceptions import TimeoutError
import datetime
from threading import Thread
import time

app = Celery('tasks')
app.config_from_object(f'src.batch_solver.celery.config')
print(app.control.inspect().active())
node_count = len(app.control.inspect().active())


class CeleryPool():
  def __enter__(self):
    return self
 
  def __exit__(self, *args):
    pass

  def __init__(self):
    ## assign empty tasks to update prefetch counts
    pass


  def assign_tasks(self):
    for i, cmd in enumerate(self.cmds):
      self.async_results[i] = self.func.s(cmd).apply_async()
      if i - self.task_cnt + len(self.task_unfinished) > max_task_queue_length:
        time.sleep(10)

  def map(self, func: Task, tasks: list) -> list:
    self.func = func
    cmds = [f"{solver_path} {task[0]} {task[1]} {task[2]}" for task in tasks]
    self.cmds = cmds
    self.task_cnt = len(self.cmds)

    # task_group = group([
    #   func.s(arg) for arg in iterable
    # ])
    # group_result :GroupResult = task_group.apply_async()
    # return group_result.get()

    self.async_results :list[AsyncResult|None]= [None] * self.task_cnt
    thread_assign_tasks = Thread(target=self.assign_tasks)
    thread_assign_tasks.start()

    pbar = tqdm.tqdm(total=self.task_cnt, smoothing=0)
    self.task_unfinished = set(range(self.task_cnt))
    last_commit = datetime.datetime.now()
    while len(self.task_unfinished) != 0:
      for i in list(self.task_unfinished):
        if self.async_results[i] is None:
          ## task not assigned yet, no need to check further for now, restart check from beginning
          break
        try:
          if self.async_results[i].ready():
            # try:
            result = self.async_results[i].get(
              # timeout=1
            )
            # except TimeoutError:
              # continue
            n, seed, _ = tasks[i]
            value, step, diff, _ = result.split("\n")
            cur.execute(f"""
              REPLACE INTO record VALUES
                ({n}, {seed}, {value}, {step}, {diff})
            """)

            self.task_unfinished.remove(i)
            pbar.update(1)
          else:
            ## task, not finished, wait for the next iteration to check readiness
            continue
        except:
          # print(f"task ready error: {tasks[i]}")
          continue
      if datetime.datetime.now() - last_commit > datetime.timedelta(minutes=1):
        con.commit()
        last_commit = datetime.datetime.now()

    thread_assign_tasks.join()
    con.commit()


# %%
total_cnts, min_cnts, min_vals, min_val_seeds, missed_tasks = get_status(ns, gen_missed_tasks=True)

while(len(missed_tasks)):
  print(f"fixing missed tasks: {len(missed_tasks)}")
  pool = CeleryPool()
  pool.map(exec_cmd, missed_tasks)
  total_cnts, min_cnts, min_vals, min_val_seeds, missed_tasks = get_status(ns, gen_missed_tasks=True)

# %%
is_to_continue = True
while is_to_continue:
  ## fix missed tasks in previous run
  while(len(missed_tasks)):
    print(f"fixing missed tasks: {len(missed_tasks)}")
    pool = CeleryPool()
    pool.map(exec_cmd, missed_tasks)
    total_cnts, min_cnts, min_vals, min_val_seeds, missed_tasks = get_status(ns, gen_missed_tasks=True)

  is_to_continue = False
  print("gen tasks bf")
  tasks = []
  for n in tqdm.tqdm(ns):
    batch_size = get_batch_size(n)
    min_cnt_n = get_dict_with_default(min_cnts, n)
    total_cnt_n = get_dict_with_default(total_cnts, n)
    if min_cnt_n < min_min_count \
      or total_cnt_n < batch_size:
        is_to_continue = True
        initial_seed = total_cnt_n
        target_total_cnt = (initial_seed//batch_size + 1) * batch_size
        if min_cnt_n:
          target_total_cnt = max(
            target_total_cnt,
            (initial_seed//min_cnt_n + 1) * min_min_count
          )
        for i in range(initial_seed, target_total_cnt):
          tasks.append((n, i, ""))
        total_cnts[n] = target_total_cnt

  pool = CeleryPool()
  pool.map(exec_cmd, tasks)

  total_cnts, min_cnts, min_vals, min_val_seeds, missed_tasks = get_status(ns, gen_missed_tasks=True)
  if len(missed_tasks):
    is_to_continue = True

print("gen tasks minimal value seed")
tasks = []
for n in tqdm.tqdm(ns):
  tasks.append((n, min_val_seeds[n], txt_dump_path))

pool = CeleryPool()
pool.map(exec_cmd, tasks)

# %%
# cur.close()
# con.commit()
# con.close()

# %%
