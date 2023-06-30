from celery import Celery
import subprocess
import os
import time
from celery.exceptions import Reject

app = Celery('tasks')
app.config_from_object('src.batch_solver.celery.config')

@app.task(bind=True, default_retry_delay=1)
def exec_cmd(self, cmd):
  try:
    return subprocess.check_output(cmd.split(" "), preexec_fn=lambda : os.nice(19)).decode()
  except:
    self.retry()

@app.task()
def do_nothing():
  pass
