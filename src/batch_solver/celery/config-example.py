## Broker settings.
broker_url = 'amqp://{user}:{password}@{address}/{database}'

# List of modules to import when the Celery worker starts.
# imports = ('myapp.tasks',)

## Using the database to store task state and results.
result_backend = 'redis://:{password}@{address}:{port}/{number}'
# result_backend = 'rpc://'

# task_annotations = {'tasks.add': {'rate_limit': '10/s'}}

# task_acks_late = True
task_reject_on_worker_lost = True
# task_acks_on_failure_or_timeout = False

worker_autoscaler = "src.batch_solver.celery.autoscaler_cpu:AutoscalerCpu"
worker_prefetch_multiplier = 1

worker_cancel_long_running_tasks_on_connection_loss = True
