from celery.worker.autoscale import Autoscaler
import logging
import os
# import psutil
import math
import datetime
import celery.concurrency.asynpool
from kombu.common import QoS
from celery.worker.consumer import Consumer


class AutoscalerCpu(Autoscaler):
  """
  Use cpu resource in such a manner that the effect to other prcsesses is minimized.
  Only physical cores count, not logical cores.
  """

  ## target system load
  CPU_LOAD_PCT_TARGET = 0.95
  ## once system load reach this, hard measure would be taken
  CPU_LOAD_PCT_HARD_THROTTLE = 0.98
  ## constant scaling mess up with the pool since current precise load is not achievable
  SCALE_MIN_INTERVAL = datetime.timedelta(seconds=3)
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.reource_monitor = ResourceMonitor()
    self.cpu_physical_cnt = self.reource_monitor.cpu_physical_cnt
    logging.info("AutoscalerCpu: cpu_physical_cnt %s", self.cpu_physical_cnt)
    ## all loads are represented in num of cpu.
    self.cpu_load_target = self.cpu_physical_cnt * self.CPU_LOAD_PCT_TARGET
    
    self.last_scale_time = datetime.datetime.now() - self.SCALE_MIN_INTERVAL
    # self.is_scaling_down = False

    # self.consumer: Consumer= self.worker.consumer
    # self.qos: QoS= self.consumer.qos

  def _maybe_scale(self, req = None):
    '''Scale up or down according to load''' 

    now = datetime.datetime.now()

    if now - self.last_scale_time >= self.SCALE_MIN_INTERVAL:
      self.last_scale_time = now

      self.process_cnt_target = self.process_active_cnt + math.floor(self.cpu_load_target - self.reource_monitor.cpu_load)
      self.process_cnt_target = max(self.min_concurrency, self.process_cnt_target)
      self.process_cnt_target = min(self.max_concurrency, self.process_cnt_target)
      ## once system load reach this, full speed to shrink pool during SCALE_MIN_INTERVAL
      if self.reource_monitor.cpu_percent > self.CPU_LOAD_PCT_HARD_THROTTLE:
        self.process_cnt_target = 1

    logging.info("AutoscalerCpu: self.process_cnt %s", self.process_cnt)
    logging.info("AutoscalerCpu: self.process_active_cnt %s", self.process_active_cnt)
    logging.info("AutoscalerCpu: self.process_cnt_target %s", self.process_cnt_target)

    logging.info("AutoscalerCpu: cpu_load_target %s", self.cpu_load_target)
    logging.info("AutoscalerCpu: cpu_load %s", self.reource_monitor.cpu_load)

    logging.info("AutoscalerCpu: cpu_load_realtime %s", self.reource_monitor.cpu_load_realtime)
    logging.info("AutoscalerCpu: cpu_load_avg_1min %s", self.reource_monitor.cpu_load_avg_1min)

    logging.info("AutoscalerCpu: cpu_percent %s", self.reource_monitor.cpu_percent)

    logging.info("AutoscalerCpu: cpu_percent_realtime %s", self.reource_monitor.cpu_percent_realtime)
    logging.info("AutoscalerCpu: cpu_percent_avg_1min %s", self.reource_monitor.cpu_percent_avg_1min)

    if self.process_cnt != self.process_cnt_target:
      self._update_consumer_prefetch_count(self.process_cnt_target)
    if self.process_cnt < self.process_cnt_target:
      self.scale_up()
      return True
    if self.process_cnt > self.process_cnt_target:
      self.scale_down()
      return True

  def scale_up(self):
    self._grow(self.process_cnt_target - self.process_cnt)

  def scale_down(self):
    self._shrink(self.process_cnt - self.process_cnt_target)
    # if self.is_scaling_down:
    #   return
    # thread_keep_scaling_down = threading.Thread(target=self._keep_scaling_down)
    # thread_keep_scaling_down.start()
  
  # def _keep_scaling_down(self):
  #   self.is_scaling_down = True
  #   while self.processes != self.processes_target:
  #     self._shrink(self.processes - self.processes_target)
  #   self.is_scaling_down = False

  @property
  def processes(self) -> int:
    if self.pool:
      return self.pool.num_processes
    else:
      return 0
    
  @property
  def process_cnt(self) -> int:
    return self.processes

  @property
  def process_active_cnt(self) -> int:
    # pool: celery.concurrency.asynpool.AsynPool = self.pool._pool
    # if type(pool) == celery.concurrency.asynpool.AsynPool:
    #   cnt = 0
    #   for worker in pool._pool:
    #     if pool._worker_active(worker):
    #       cnt += 1
    #   return cnt
    return self.process_cnt


import threading
import psutil
import socket
import os

class ResourceMonitor():
  def __init__(
    self,
    watch_cpu_itv = 1,
  ):
    self.watch_cpu_itv = watch_cpu_itv
    self.cpu_physical_cnt = psutil.cpu_count(logical = False)
    self.cpu_logical_cnt = psutil.cpu_count(logical = True)

    if socket.gethostname() == "zju109092050wu":
      self.cpu_physical_cnt = 48
    
    self._gen_cpu_percent()

    thread_watch_cpu_percent = threading.Thread(target=self._watch_cpu_percent)
    thread_watch_cpu_percent.start()

  def _gen_cpu_percent(self):
    self.cpu_percent_realtime = psutil.cpu_percent(self.watch_cpu_itv)/100
    self.cpu_load_realtime = self.cpu_percent_realtime * self.cpu_logical_cnt
    self.cpu_load_avg_1min = os.getloadavg()[0]
    self.cpu_percent_avg_1min = self.cpu_load_avg_1min / self.cpu_logical_cnt
    self.cpu_percent = max(self.cpu_percent_realtime, self.cpu_percent_avg_1min)
    self.cpu_load = max(self.cpu_load_realtime, self.cpu_load_avg_1min)

  def _watch_cpu_percent(self):
    while True:
      self._gen_cpu_percent()
