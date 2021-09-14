"""Common configurations."""
from lingvo.core import base_model_params
from lingvo.core import hyperparams
from lingvo.core import program as program_lib
from lingvo.core import schedule
import numpy as np

from google3.learning.brain.research.babelfish import model_helper
from google3.learning.brain.research.babelfish.multimodal import cluster_utils


def Global2PerSplitBatchSize(global_batch_size: int) -> int:
  """Converts global batch size to per-split batch size."""
  num_splits = cluster_utils.GetNumSplitsPerClient()
  if global_batch_size % num_splits != 0:
    raise ValueError(f'Global batch size {global_batch_size} cannot evenly '
                     f'distributes to {num_splits} splits.')
  per_split_batch_size = global_batch_size // num_splits
  return per_split_batch_size


def ConfigureInput(
    p: hyperparams.Params,
    is_training: bool,
    global_batch_size: int,
    num_batcher_threads: int = 128,
    file_parallelism: int = 128,
    file_buffer_size: int = 5e4,
    file_random_seed: int = 0,
    num_partitions: int = 1,
    use_per_host_infeed: bool = True,
) -> hyperparams.Params:
  """Commonly used input configurations."""
  # When is_training is False, we set both use_per_host_infeed and
  # use_per_core_infeed to be False so that eval can be done on the test data
  # for exactly one epoch.
  p.use_per_host_infeed = use_per_host_infeed and is_training
  p.use_per_core_infeed = False
  p.file_random_seed = file_random_seed
  p.num_batcher_threads = num_batcher_threads
  p.file_parallelism = file_parallelism
  p.file_buffer_size = file_buffer_size
  p.batch_size = Global2PerSplitBatchSize(global_batch_size)

  if num_partitions > 1:
    p = model_helper.ChangeToGShardInput(
        p,
        num_partitions=num_partitions,
        scale_batch_size=False,
        is_eval=not is_training)
    # Disable partitioned_infeed_queue as otherwise eval programs won't run.
    p.use_partitioned_infeed_queue = False
  return p


def CosineLRSchedule(warmup_steps: int,
                     total_steps: int,
                     final_value: int = 0.0):
  """Returns lr schedule as linear warmup + cosine decaying."""
  return schedule.PiecewiseSchedule.Params().Set(
      boundaries=[warmup_steps],
      schedules=[
          schedule.LinearSchedule.Params().Set(
              start=(0, 0.0),
              limit=(warmup_steps, 1.0),
          ),
          schedule.CosineSchedule.Params().Set(
              total_steps=total_steps - warmup_steps,
              final_value=final_value,
          ),
      ])


def LinearLRSchedule(warmup_steps: int,
                     total_steps: int,
                     final_value: int = 0.0):
  """Returns lr schedule as linear warmup + linear decaying."""
  return schedule.PiecewiseSchedule.Params().Set(
      boundaries=[warmup_steps],
      schedules=[
          schedule.LinearSchedule.Params().Set(
              start=(0, 0.0),
              limit=(warmup_steps, 1.0),
          ),
          schedule.LinearSchedule.Params().Set(
              start=(0, 1.),
              limit=(total_steps - warmup_steps, final_value),
          ),
      ])


def ExponentialLRSchedule(warmup_steps: int,
                          decay_start_steps: int,
                          total_steps: int,
                          final_value: int = 0.0):
  """Returns lr schedule as linear warmup + exponential decaying."""
  return schedule.LinearRampupExponentialDecay.Params().Set(
      warmup=warmup_steps,
      decay_start=decay_start_steps,
      decay_end=total_steps,
      min=final_value)


class BaseTask(base_model_params.SingleTaskModelParams):
  """A template base task with commonly used configurations."""

  # Set input hparams.
  TRAIN_BATCH_SIZE = 4096
  EVAL_BATCH_SIZE = 128

  # Set optimization hparams.
  CLIP_GRADIENT_NORM_TO_VALUE = 0.0
  LEARNING_RATE = 0.0
  NUM_EPOCHS = 1
  L2_REGULARIZER_WEIGHT = 0.0
  EMA_DECAY = 0.0

  # GSPMD-related device mesh shape used to calculate device mesh.
  DEVICE_MESH_SHAPE = [1, 1]

  def _NumDevices(self):
    return np.prod(self.DEVICE_MESH_SHAPE)

  def _DeviceMesh(self):
    if self.DEVICE_MESH_SHAPE[0] == 1 and self.DEVICE_MESH_SHAPE[1] == 1:
      return None  # Use tf replica.
    else:
      return np.reshape(np.arange(self._NumDevices()), self.DEVICE_MESH_SHAPE)

  def _NumStepsPerEpoch(self):
    return self._NumSamplesPerEpoch() // self.TRAIN_BATCH_SIZE

  def _NumSamplesPerEpoch(self):
    return self.Train().num_samples

  def _LearningRateScheduleParams(self):
    """Returns learning rate schedule params for the task."""
    raise NotImplementedError('Abstract method')

  def _OptimizerParams(self):
    """Returns optimizer params for the task."""
    raise NotImplementedError('Abstract method')

  def _ConfigureTask(self, p: hyperparams.Params) -> hyperparams.Params:
    """Configures commonly used p.eval and p.train."""
    # Evaluate the whole set.
    p.eval.samples_per_summary = 0

    # Set checkpointing.
    tp = p.train
    tp.save_max_to_keep = 10
    tp.save_keep_checkpoint_every_n_hours = 0.25
    tp.save_interval_seconds = 60  # More frequent checkpoints.
    tp.tpu_steps_per_loop = 1000

    # Set optimizer.
    tp.optimizer = self._OptimizerParams()
    tp.lr_schedule = self._LearningRateScheduleParams()
    tp.learning_rate = self.LEARNING_RATE
    tp.max_steps = self.NUM_EPOCHS * self._NumStepsPerEpoch()

    # Set regularization.
    tp.l2_regularizer_weight = self.L2_REGULARIZER_WEIGHT
    tp.clip_gradient_norm_to_value = self.CLIP_GRADIENT_NORM_TO_VALUE

    # Set exponential moving average decay.
    if self.EMA_DECAY != 0:
      tp.ema_decay_moving_vars = True
      tp.ema_decay = self.EMA_DECAY

    # Set model loading rules if presented.
    if hasattr(self, 'CHECKPOINT_PATH') and hasattr(self, 'OVERRIDE_RULES'):
      tp.init_from_checkpoint_rules = {
          self.CHECKPOINT_PATH: (self.OVERRIDE_RULES,
                                 getattr(self, 'IGNORE_RULES', []))
      }
    return p

  def _ConfigureTaskMixed(self, p: hyperparams.Params) -> hyperparams.Params:
    """Configures commonly used p.eval and p.train."""
    # Evaluate the whole set.
    p.eval.samples_per_summary = 0

    # Set checkpointing.
    tp = p.train
    tp.save_max_to_keep = 10
    tp.save_keep_checkpoint_every_n_hours = 0.25
    tp.save_interval_seconds = 60  # More frequent checkpoints.
    tp.tpu_steps_per_loop = 1000

    # Set optimizer.
    tp.optimizer = self._OptimizerParams()
    tp.lr_schedule = self._LearningRateScheduleParams()
    tp.learning_rate = self.LEARNING_RATE
    tp.max_steps = self.NUM_EPOCHS * self._NumStepsPerEpoch()

    # Set regularization.
    tp.l2_regularizer_weight = self.L2_REGULARIZER_WEIGHT
    tp.clip_gradient_norm_to_value = self.CLIP_GRADIENT_NORM_TO_VALUE

    # Set exponential moving average decay.
    if self.EMA_DECAY != 0:
      tp.ema_decay_moving_vars = True
      tp.ema_decay = self.EMA_DECAY

    # Set model loading rules if presented.
    if hasattr(self, 'CHECKPOINT_PATH') and hasattr(self, 'OVERRIDE_RULES')\
        and hasattr(self, 'CHECKPOINT_PATH_ex')\
        and hasattr(self, 'OVERRIDE_RULES_ex'):
      tp.init_from_checkpoint_rules = {
          self.CHECKPOINT_PATH: (self.OVERRIDE_RULES,
                                 getattr(self, 'IGNORE_RULES', [])),
          self.CHECKPOINT_PATH_ex: (self.OVERRIDE_RULES_ex,
                                 getattr(self, 'IGNORE_RULES_ex', []))
      }
    return p

  def ProgramSchedule(self):
    """Configures commonly used program schedule."""
    eval_steps_per_loop = self.Test().num_samples // self.EVAL_BATCH_SIZE + 1
    p = program_lib.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=1000,
        eval_dataset_names=['Test'],
        eval_steps_per_loop=eval_steps_per_loop,
        decode_steps_per_loop=eval_steps_per_loop,
    )
    # Evaluate test set per 5k iterations.
    p.train_executions_per_eval = 5
    return p
