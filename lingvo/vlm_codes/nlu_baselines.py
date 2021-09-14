"""Params of natural language understanding (NLU) tasks."""
import os

from lingvo.core import layers as lingvo_layers
from lingvo.core import learner
from lingvo.core import program as program_lib
from lingvo.core import py_utils

from google3.learning.brain.research.babelfish import model_helper
from google3.learning.brain.research.babelfish import model_registry
from google3.learning.brain.research.babelfish import optimizer as bf_optimizer
from google3.learning.brain.research.babelfish.lm import t5_tokenizer
from google3.learning.brain.research.babelfish.mt import base_config
from google3.learning.brain.research.babelfish.multimodal import cluster_utils
from google3.learning.brain.research.babelfish.multimodal import datasets
from google3.learning.brain.research.babelfish.multimodal import mt_decoder
from google3.learning.brain.research.babelfish.multimodal import mt_encoder
from google3.learning.brain.research.babelfish.multimodal import tasks
from google3.learning.brain.research.babelfish.multimodal.params import common


class GLUEClassificationTemplate(common.BaseTask):
  """GLUE classification task with an encoder-decoder model."""

  # TODO(ziruiw): consolidate params when moving out of the experimental
  # CHECKPOINT_PATH = ('/cns/nm-d/home/jiahuiyu/brain/rs=6.3/ImageText2TextLMStochasticDepthLarge.t10/train')
  # OVERRIDE_RULES = [('GLUETask/(.*/var:0$)', 'ImageText2TextLMTask/%s')]

  # CHECKPOINT_PATH = ('/cns/jn-d/home/ziruiw/brain/rs=6.3/ImageText2TextLM.small.ibz4096.tbz512.PrefixLM.Res50.Trans2.BatchMajor.RelPos.LR5e4.WD1e2/train/')
  # OVERRIDE_RULES = [('GLUETask/(.*/var:0$)', 'ImageText2TextLMTask/%s')]

  CHECKPOINT_PATH = ('/cns/tp-d/home/runzheyang/brain/rs=6.3/text2textlm.small.fixedtranspose.1/train/')
  OVERRIDE_RULES = [('GLUETask/(.*/var:0$)', 'Text2TextLM/%s')]

  # CHECKPOINT_PATH = ('/cns/tp-d/home/runzheyang/brain/rs=6.3/text2textlm.small.fixedtranspose.200k/train/')
  # OVERRIDE_RULES = [('GLUETask/(.*/var:0$)', 'Text2TextLM/%s')]

  # CHECKPOINT_PATH = ('/cns/tp-d/home/runzheyang/brain/rs=6.3/text2textlm.small.notranspose.200k/train/')
  # OVERRIDE_RULES = [('GLUETask/(.*/var:0$)', 'Text2TextLM/%s')]

  # CHECKPOINT_PATH = ('/cns/tp-d/home/runzheyang/brain/rs=6.3/text2textlm.base/train/')
  # OVERRIDE_RULES = [('GLUETask/(.*/var:0$)', 'Text2TextLM/%s')]

  # CHECKPOINT_PATH = ('/cns/tp-d/home/runzheyang/brain/rs=6.3/text2textlm.large.lr5e-5/train/')
  # OVERRIDE_RULES = [('GLUETask/(.*/var:0$)', 'Text2TextLM/%s')]


  IGNORE_RULES = [
      'GLUETask/output_projection/(.*/var:0$)', 'GLUETask/softmax/(.*/var:0$)',
      'GLUETask/projector_ln/(.*/var:0$)'
  ]
  CLASSIFIER_MUL = 10.
  # CLASSIFIER_MUL = 5.

  # Training param
  MAX_STEPS = 10000
  # MAX_STEPS = 20000
  LEARNING_RATE = 3e-5
  CLIP_GRADIENT_NORM_TO_VALUE = 0.0
  TRAIN_BATCH_SIZE = 128
  EVAL_BATCH_SIZE = 128
  DROPOUT_RATE = 0.1

  # Model param (SMALL)
  NUM_LAYERS = 8
  NUM_HEADS = 8
  MODEL_DIM = 512
  MLP_DIM = MODEL_DIM * 4
  TEXT_VOCAB_SIZE = 32000
  SHARE_EMBED = True


  # Model param (BASE)
  # NUM_LAYERS = 12
  # NUM_HEADS = 12
  # MODEL_DIM = 768
  # MLP_DIM = MODEL_DIM * 4
  # TEXT_VOCAB_SIZE = 32000
  # SHARE_EMBED = True


  #  # Model param (LARGE)
  # NUM_LAYERS = 24
  # NUM_HEADS = 16
  # MODEL_DIM = 1024
  # MLP_DIM = MODEL_DIM * 4
  # TEXT_VOCAB_SIZE = 32000
  # SHARE_EMBED = True

  # Task param
  GLUE_TASK_NAME = ''
  LABEL_NAMES = []
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 0
  NUM_EVAL_SAMPLES = 0
  EVAL_DATASET_NAME = 'validation'

  def _OptimizerParams(self):
    return bf_optimizer.AdamW.Params().Set(
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-6,
        weight_decay=0.1,
    )

  def _LearningRateScheduleParams(self):
    return common.LinearLRSchedule(
        warmup_steps=self.MAX_STEPS // 10,
        total_steps=self.MAX_STEPS
        )

  def ProgramSchedule(self):
    eval_steps_per_loop = self.Test().num_samples // self.EVAL_BATCH_SIZE + 1
    p = program_lib.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=100,
        eval_dataset_names=['Test'],
        eval_steps_per_loop=eval_steps_per_loop,
        decode_steps_per_loop=0,
    )

    # Add decoding for F1 and MCC scores.
    decode_program_params = program_lib.DecodeProgram.Params()
    decode_program_params.name = 'decode_tpu'
    decode_program_params.steps_per_loop = eval_steps_per_loop
    decode_program_params.dataset_name = 'Test'
    p.eval_programs.append(decode_program_params)

    p.train_executions_per_eval = 1
    return p

  def EncoderParams(self):
    p = base_config.SetupTransformerBatchMajorEncoderV2(
        self.MODEL_DIM,
        self.TEXT_VOCAB_SIZE,
        self.NUM_LAYERS,
        self.NUM_HEADS,
        self.MLP_DIM,
        residual_dropout_prob=self.DROPOUT_RATE,
        input_dropout_prob=self.DROPOUT_RATE,
        atten_dropout_prob=self.DROPOUT_RATE,
        relu_dropout_prob=self.DROPOUT_RATE,
        add_unnormalized_residuals=True,
        enable_per_dim_scale=False,
    )
    p = mt_encoder.TransformerBatchMajorEncoderSeparateFprop.Params().Set(
        **dict(p.Copy().IterParams()))
    p.cls = mt_encoder.TransformerBatchMajorEncoderSeparateFprop
    p.output_data_format = 'TBC'
    return p

  def DecoderParams(self):
    p = base_config.SetupTransformerBatchMajorDecoderV2(
        self.MODEL_DIM,
        self.TEXT_VOCAB_SIZE,
        self.NUM_LAYERS,
        self.NUM_HEADS,
        self.MLP_DIM,
        residual_dropout_prob=self.DROPOUT_RATE,
        input_dropout_prob=self.DROPOUT_RATE,
        atten_dropout_prob=self.DROPOUT_RATE,
        relu_dropout_prob=self.DROPOUT_RATE,
        label_smoothing_uncertainty=0.0,
        add_unnormalized_residuals=True,
        enable_per_dim_scale=False,
    )
    p = mt_decoder.PrefixTransformerBatchMajorDecoder.Params().Set(
        **dict(p.Copy().IterParams()))
    p.cls = mt_decoder.PrefixTransformerBatchMajorDecoder
    p.input_data_format = 'TBC'
    return p

  def ConfigureSequenceInput(self, p, is_training, global_batch_size):
    p.use_per_core_infeed = False
    p.use_per_host_infeed = is_training
    p.file_random_seed = 0
    p.num_batcher_threads = 16 if is_training else 1
    p.file_parallelism = 16
    p.file_buffer_size = 10000

    num_splits = cluster_utils.GetNumSplitsPerClient()

    if global_batch_size % num_splits != 0:
      raise ValueError(f'Global batch size {global_batch_size} cannot evenly '
                       f'distributes to {num_splits} splits.')
    p.bucket_batch_limit = [global_batch_size // num_splits]
    return p

  def _SetLearner(self, p):

    classifier_learner = learner.ExtractLearnerFromLegacyParams(p.train).Set(
        name='classifier',
        loss_name='loss',
        learning_rate=self.CLASSIFIER_MUL * self.LEARNING_RATE,
        bprop_variable_filter=('/output_projection/'
                               '|/projector_ln/'
                               '|GLUETask/softmax/'
                               '|GLUETask/shared_emb/'))

    pretrained_learner = classifier_learner.Copy().Set(
        name='pretrained',
        loss_name='loss',
        learning_rate=self.LEARNING_RATE,
        bprop_variable_exclusion=classifier_learner.bprop_variable_filter,
        bprop_variable_filter=None)

    p.train.learner = [classifier_learner, pretrained_learner]


  def Train(self):
    p = datasets.glue_task(
        mixture_or_task_name=self.GLUE_TASK_NAME,
        label_names=self.LABEL_NAMES,
        split='train')
    p.num_samples = self.NUM_TRAIN_SAMPLES
    p.tokenizer = t5_tokenizer.T5Tokenizer.Params().Set(
        spm_model=os.path.join('/placer/prod/home/brain-ogm-data-writer',
                               'spm/cc_all.32000/sentencepiece.model'),
        target_sos_id=0,
        target_eos_id=1,
        target_unk_id=2)
    p = self.ConfigureSequenceInput(
        p, is_training=True, global_batch_size=self.TRAIN_BATCH_SIZE)
    return p

  def Test(self):
    p = datasets.glue_task(
        mixture_or_task_name=self.GLUE_TASK_NAME,
        label_names=self.LABEL_NAMES,
        split=self.EVAL_DATASET_NAME)
    p.num_samples = self.NUM_EVAL_SAMPLES
    p.tokenizer = t5_tokenizer.T5Tokenizer.Params().Set(
        spm_model=os.path.join('/placer/prod/home/brain-ogm-data-writer',
                               'spm/cc_all.32000/sentencepiece.model'),
        target_sos_id=0,
        target_eos_id=1,
        target_unk_id=2)
    p = self.ConfigureSequenceInput(
        p, is_training=False, global_batch_size=self.EVAL_BATCH_SIZE)
    return p

  def Task(self):
    p = tasks.GLUEClassification.Params().Set(
        name='GLUETask', num_classes=self.NUM_LABELS)

    p.encoder = self.EncoderParams()
    p.decoder = self.DecoderParams()
    p.projector = lingvo_layers.FCLayer.Params().Set(
        name='output_projection',
        input_dim=self.MODEL_DIM,
        output_dim=self.MODEL_DIM * 2,
        activation='GELU',
    )
    p.softmax = lingvo_layers.SimpleFullSoftmax.Params().Set(
        params_init=py_utils.WeightInit.Gaussian(scale=0.01),
        input_dim=self.MODEL_DIM * 2,
        num_classes=self.NUM_LABELS,
    )

    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)
    sm_params = p.decoder.softmax.Copy()
    sm_params.input_dim = self.MODEL_DIM
    shared_emb = lingvo_layers.SharedSoftmaxLayer.Params().Set(
        softmax=sm_params, num_classes=sm_params.num_classes)
    shared_emb.scale_sqrt_depth = True
    shared_emb.softmax.use_num_classes_major_weight = False
    p.decoder.shared_emb = shared_emb
    p.encoder.shared_emb = shared_emb

    p = self._ConfigureTask(p)

    self._SetLearner(p)
    p.train.max_steps = self.MAX_STEPS

    return p

@model_registry.RegisterSingleTaskModel
class CoLAClassification(GLUEClassificationTemplate):
  """Finetune CoLA task from a pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  small it2t:
  """
  GLUE_TASK_NAME = 'glue_cola_v002'
  LABEL_NAMES = ['unacceptable', 'acceptable']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 8551
  NUM_EVAL_SAMPLES = 1043

  # Training param
  TRAIN_BATCH_SIZE = 128
  EVAL_BATCH_SIZE = 128
  LEARNING_RATE = 1e-6


@model_registry.RegisterSingleTaskModel
class SST2Classification(GLUEClassificationTemplate):
  """Finetune SST-2 task from a pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  small it2t: go/mldash/5214407889154626065
  """
  GLUE_TASK_NAME = 'glue_sst2_v002'
  LABEL_NAMES = ['negative', 'positive']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 67349
  NUM_EVAL_SAMPLES = 872

  # Training param
  # LEARNING_RATE = 5e-5
  # LEARNING_RATE = 3e-5
  LEARNING_RATE = 5e-6
  TRAIN_BATCH_SIZE = 512
  EVAL_BATCH_SIZE = 128


@model_registry.RegisterSingleTaskModel
class MRPCClassification(GLUEClassificationTemplate):
  """Finetune MRPC task from a pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  """
  GLUE_TASK_NAME = 'glue_mrpc_v002'
  LABEL_NAMES = ['not_equivalent', 'equivalent']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 3668
  NUM_EVAL_SAMPLES = 408
  LEARNING_RATE = 8e-5


@model_registry.RegisterSingleTaskModel
class QQPClassification(GLUEClassificationTemplate):
  """Finetune QQP task from a pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  """
  GLUE_TASK_NAME = 'glue_qqp_v002'
  LABEL_NAMES = ['not_duplicate', 'duplicate']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 363846
  NUM_EVAL_SAMPLES = 40430
  LEARNING_RATE = 3e-5


@model_registry.RegisterSingleTaskModel
class QNLIClassification(GLUEClassificationTemplate):
  """Finetune QNLI task from a pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  """
  GLUE_TASK_NAME = 'glue_qnli_v002'
  LABEL_NAMES = ['not_entailment', 'entailment']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 104743
  NUM_EVAL_SAMPLES = 5463
  LEARNING_RATE = 3e-5


@model_registry.RegisterSingleTaskModel
class WNLIClassification(GLUEClassificationTemplate):
  """Finetune WNLI task from a pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  """
  GLUE_TASK_NAME = 'glue_wnli_v002'
  LABEL_NAMES = ['not_entailment', 'entailment']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 635
  NUM_EVAL_SAMPLES = 71
  LEARNING_RATE = 3e-5


@model_registry.RegisterSingleTaskModel
class RTEClassification(GLUEClassificationTemplate):
  """Finetune RTE task from a pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  """
  GLUE_TASK_NAME = 'glue_rte_v002'
  LABEL_NAMES = ['not_entailment', 'entailment']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 2490
  NUM_EVAL_SAMPLES = 277

  # Training param
  TRAIN_BATCH_SIZE = 128
  EVAL_BATCH_SIZE = 128
  LEARNING_RATE = 5e-6


@model_registry.RegisterSingleTaskModel
class MNLImClassification(GLUEClassificationTemplate):
  """Finetune MNLI-m task from a pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  """
  GLUE_TASK_NAME = 'glue_mnli_v002'
  LABEL_NAMES = ['contradiction', 'neutral', 'entailment']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 392702
  NUM_EVAL_SAMPLES = 9815
  EVAL_DATASET_NAME = 'test_matched'
  LEARNING_RATE = 3e-5

@model_registry.RegisterSingleTaskModel
class MNLImmClassification(GLUEClassificationTemplate):
  """Finetune MNLI-mm task from a pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  """
  GLUE_TASK_NAME = 'glue_mnli_v002'
  LABEL_NAMES = ['contradiction', 'neutral', 'entailment']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 392702
  NUM_EVAL_SAMPLES = 9832
  EVAL_DATASET_NAME = 'test_mismatched'
  LEARNING_RATE = 3e-5

class MixedFinetuneTemplate(common.BaseTask):
  """Mixed Finetune task with T2T and IT2T encoder-decoder model."""

  CHECKPOINT_PATH_ex = ('/cns/mb-d/home/yuancao/brain/rs=6.3/mm_it2t_10_0.5/train/')
  OVERRIDE_RULES_ex = [('MixedFinetune/encoder_ex/(.*/var:0$)', 'ImageText2TextLMTask/encoder/%s'),
                       ('MixedFinetune/decoder_ex/(.*/var:0$)', 'ImageText2TextLMTask/decoder/%s'),
                       ('MixedFinetune/shared_emb_ex/(.*/var:0$)', 'ImageText2TextLMTask/shared_emb/%s')]
  IGNORE_RULES_ex = [
      'MixedFinetune/output_projection/(.*/var:0$)', 'MixedFinetune/softmax/(.*/var:0$)',
      'MixedFinetune/projector_ln/(.*/var:0$)', 'MixedFinetune/encoder/(.*/var:0$)',
      'MixedFinetune/decoder/(.*/var:0$)', 'MixedFinetune/embedding_remap/(.*/var:0$)',
      'MixedFinetune/decoder_ex/softmax/(.*/var:0$)',
      'MixedFinetune/embedding_selector/(.*/var:0$)', 'MixedFinetune/shared_emb/(.*/var:0$)',
  ]
  # CHECKPOINT_PATH_ex = ('/cns/jn-d/home/ziruiw/brain/rs=6.3/ImageText2TextLM.small.ibz4096.tbz512.PrefixLM.Res50.Trans2.BatchMajor.RelPos.LR5e4.WD1e2/train/')
  # OVERRIDE_RULES_ex = [('MixedFinetune/encoder_ex/(.*/var:0$)', 'ImageText2TextLMTask/encoder/%s'),
  #                      ('MixedFinetune/decoder_ex/(.*/var:0$)', 'ImageText2TextLMTask/decoder/%s'),
  #                      ('MixedFinetune/shared_emb_ex/(.*/var:0$)', 'ImageText2TextLMTask/shared_emb/%s')]
  # IGNORE_RULES_ex = [
  #     'MixedFinetune/output_projection/(.*/var:0$)', 'MixedFinetune/softmax/(.*/var:0$)',
  #     'MixedFinetune/projector_ln/(.*/var:0$)', 'MixedFinetune/encoder/(.*/var:0$)',
  #     'MixedFinetune/decoder/(.*/var:0$)', 'MixedFinetune/embedding_remap/(.*/var:0$)',
  #     'MixedFinetune/decoder_ex/softmax/(.*/var:0$)',
  #     'MixedFinetune/embedding_selector/(.*/var:0$)', 'MixedFinetune/shared_emb/(.*/var:0$)',
  # ]
  # CHECKPOINT_PATH_ex = ('/cns/tp-d/home/runzheyang/brain/rs=6.3/text2textlm.small.fixedtranspose.1.twin/train/')
  # OVERRIDE_RULES_ex = [('MixedFinetune/encoder_ex/(.*/var:0$)', 'Text2TextLM/encoder/%s'),
  #                      ('MixedFinetune/decoder_ex/(.*/var:0$)', 'Text2TextLM/decoder/%s'),
  #                      ('MixedFinetune/shared_emb_ex/(.*/var:0$)', 'Text2TextLM/shared_emb/%s')]
  # IGNORE_RULES_ex = [
  #     'MixedFinetune/output_projection/(.*/var:0$)', 'MixedFinetune/softmax/(.*/var:0$)',
  #     'MixedFinetune/projector_ln/(.*/var:0$)', 'MixedFinetune/encoder/(.*/var:0$)',
  #     'MixedFinetune/decoder/(.*/var:0$)', 'MixedFinetune/embedding_remap/(.*/var:0$)',
  #     # 'MixedFinetune/decoder_ex/softmax/(.*/var:0$)',
  #     'MixedFinetune/embedding_selector/(.*/var:0$)', 'MixedFinetune/shared_emb/(.*/var:0$)',
  # ]

  # CHECKPOINT_PATH_IT2T = ('/cns/mb-d/home/yuancao/brain/rs=6.3/mm_it2t_10_0.5/train/')
  # OVERRIDE_RULES_IT2T = [('MixedFinetune_IT2T/(.*/var:0$)', 'ImageText2TextLMTask/%s')]
  # IGNORE_RULES_IT2T = [
  #     'MixedFinetune_IT2T/output_projection/(.*/var:0$)', 'MixedFinetune_IT2T/softmax/(.*/var:0$)',
  #     'MixedFinetune_IT2T/projector_ln/(.*/var:0$)'
  # ]

  CHECKPOINT_PATH = ('/cns/tp-d/home/runzheyang/brain/rs=6.3/text2textlm.small.fixedtranspose.1/train/')
  OVERRIDE_RULES = [('MixedFinetune/(.*/var:0$)', 'Text2TextLM/%s')]
  IGNORE_RULES = [
      'MixedFinetune/output_projection/(.*/var:0$)', 'MixedFinetune/softmax/(.*/var:0$)',
      'MixedFinetune/projector_ln/(.*/var:0$)', 'MixedFinetune/encoder_ex/(.*/var:0$)',
      'MixedFinetune/decoder_ex/(.*/var:0$)', 'MixedFinetune/embedding_remap/(.*/var:0$)',
      'MixedFinetune/embedding_selector/(.*/var:0$)', 'MixedFinetune/shared_emb_ex/(.*/var:0$)',
  ]

  CLASSIFIER_MUL = 10.

  # Training param
  MAX_STEPS = 10000
  LEARNING_RATE = 3e-5
  CLIP_GRADIENT_NORM_TO_VALUE = 0.0
  TRAIN_BATCH_SIZE = 128
  EVAL_BATCH_SIZE = 128
  DROPOUT_RATE = 0.1

  # Model param (SMALL)
  NUM_LAYERS = 8
  NUM_HEADS = 8
  MODEL_DIM = 512
  MLP_DIM = MODEL_DIM * 4
  TEXT_VOCAB_SIZE = 32000
  SHARE_EMBED = True

  # Task param
  GLUE_TASK_NAME = ''
  LABEL_NAMES = []
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 0
  NUM_EVAL_SAMPLES = 0
  EVAL_DATASET_NAME = 'validation'

  def _OptimizerParams(self):
    return bf_optimizer.AdamW.Params().Set(
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-6,
        weight_decay=0.1,
    )

  def _LearningRateScheduleParams(self):
    return common.LinearLRSchedule(
        warmup_steps=self.MAX_STEPS // 10,
        total_steps=self.MAX_STEPS
        )

  def ProgramSchedule(self):
    eval_steps_per_loop = self.Test().num_samples // self.EVAL_BATCH_SIZE + 1
    p = program_lib.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=100,
        eval_dataset_names=['Test'],
        eval_steps_per_loop=eval_steps_per_loop,
        decode_steps_per_loop=0,
    )

    # Add decoding for F1 and MCC scores.
    decode_program_params = program_lib.DecodeProgram.Params()
    decode_program_params.name = 'decode_tpu'
    decode_program_params.steps_per_loop = eval_steps_per_loop
    decode_program_params.dataset_name = 'Test'
    p.eval_programs.append(decode_program_params)

    p.train_executions_per_eval = 1
    return p

  def EncoderParams(self):
    p = base_config.SetupTransformerBatchMajorEncoderV2(
        self.MODEL_DIM,
        self.TEXT_VOCAB_SIZE,
        self.NUM_LAYERS,
        self.NUM_HEADS,
        self.MLP_DIM,
        residual_dropout_prob=self.DROPOUT_RATE,
        input_dropout_prob=self.DROPOUT_RATE,
        atten_dropout_prob=self.DROPOUT_RATE,
        relu_dropout_prob=self.DROPOUT_RATE,
        add_unnormalized_residuals=True,
        enable_per_dim_scale=False,
    )
    p = mt_encoder.TransformerBatchMajorEncoderMixedSeparateFprop.Params().Set(
        **dict(p.Copy().IterParams()))
    p.cls = mt_encoder.TransformerBatchMajorEncoderMixedSeparateFprop
    p.output_data_format = 'TBC'
    return p

  def DecoderParams(self):
    p = base_config.SetupTransformerBatchMajorDecoderV2(
        self.MODEL_DIM,
        self.TEXT_VOCAB_SIZE,
        self.NUM_LAYERS,
        self.NUM_HEADS,
        self.MLP_DIM,
        residual_dropout_prob=self.DROPOUT_RATE,
        input_dropout_prob=self.DROPOUT_RATE,
        atten_dropout_prob=self.DROPOUT_RATE,
        relu_dropout_prob=self.DROPOUT_RATE,
        label_smoothing_uncertainty=0.0,
        add_unnormalized_residuals=True,
        enable_per_dim_scale=False,
    )
    p = mt_decoder.PrefixTransformerBatchMajorDecoderMixed.Params().Set(
        **dict(p.Copy().IterParams()))
    p.cls = mt_decoder.PrefixTransformerBatchMajorDecoderMixed
    p.input_data_format = 'TBC'
    return p

  def ConfigureSequenceInput(self, p, is_training, global_batch_size):
    p.use_per_core_infeed = False
    p.use_per_host_infeed = is_training
    p.file_random_seed = 0
    p.num_batcher_threads = 16 if is_training else 1
    p.file_parallelism = 16
    p.file_buffer_size = 10000

    num_splits = cluster_utils.GetNumSplitsPerClient()

    if global_batch_size % num_splits != 0:
      raise ValueError(f'Global batch size {global_batch_size} cannot evenly '
                       f'distributes to {num_splits} splits.')
    p.bucket_batch_limit = [global_batch_size // num_splits]
    return p

  def _SetLearner(self, p):

    classifier_learner = learner.ExtractLearnerFromLegacyParams(p.train).Set(
        name='classifier',
        loss_name='loss',
        learning_rate=self.CLASSIFIER_MUL * self.LEARNING_RATE,
        bprop_variable_filter=('/output_projection/'
                               '|/projector_ln/'
                               '|MixedFinetune/softmax/'))

    pretrained_learner = classifier_learner.Copy().Set(
        name='pretrained',
        loss_name='loss',
        learning_rate=self.LEARNING_RATE,
        bprop_variable_exclusion=('/output_projection/'
                                  '|/projector_ln/'
                                  '|MixedFinetune/softmax/'
                                  '|MixedFinetune/shared_emb/'
                                  '|MixedFinetune/shared_emb_ex/'),
        bprop_variable_filter=None)

    p.train.learner = [classifier_learner, pretrained_learner]

  def Train(self):
    p = datasets.glue_task(
        mixture_or_task_name=self.GLUE_TASK_NAME,
        label_names=self.LABEL_NAMES,
        split='train')
    p.num_samples = self.NUM_TRAIN_SAMPLES
    p.tokenizer = t5_tokenizer.T5Tokenizer.Params().Set(
        spm_model=os.path.join('/placer/prod/home/brain-ogm-data-writer',
                               'spm/cc_all.32000/sentencepiece.model'),
        target_sos_id=0,
        target_eos_id=1,
        target_unk_id=2)
    p = self.ConfigureSequenceInput(
        p, is_training=True, global_batch_size=self.TRAIN_BATCH_SIZE)
    return p

  def Test(self):
    p = datasets.glue_task(
        mixture_or_task_name=self.GLUE_TASK_NAME,
        label_names=self.LABEL_NAMES,
        split=self.EVAL_DATASET_NAME)
    p.num_samples = self.NUM_EVAL_SAMPLES
    p.tokenizer = t5_tokenizer.T5Tokenizer.Params().Set(
        spm_model=os.path.join('/placer/prod/home/brain-ogm-data-writer',
                               'spm/cc_all.32000/sentencepiece.model'),
        target_sos_id=0,
        target_eos_id=1,
        target_unk_id=2)
    p = self.ConfigureSequenceInput(
        p, is_training=False, global_batch_size=self.EVAL_BATCH_SIZE)
    return p

  def Task(self):
    p = tasks.MixedFinetune.Params().Set(
        name='MixedFinetune', num_classes=self.NUM_LABELS)

    p.encoder = self.EncoderParams()
    p.decoder = self.DecoderParams()
    p.encoder_ex = self.EncoderParams()
    p.decoder_ex = self.DecoderParams()
    p.emb_remap = lingvo_layers.FCLayer.Params().Set(
        name='embedding_remap',
        input_dim=self.MODEL_DIM,
        output_dim=self.MODEL_DIM,
    )
    p.emb_selector = lingvo_layers.FCLayer.Params().Set(
        name='embedding_selector',
        input_dim=self.MODEL_DIM * 2,
        output_dim=1,
        activation='SIGMOID',
    )
    p.projector = lingvo_layers.FCLayer.Params().Set(
        name='output_projection',
        input_dim=self.MODEL_DIM,
        output_dim=self.MODEL_DIM * 2,
        activation='GELU',
    )
    p.softmax = lingvo_layers.SimpleFullSoftmax.Params().Set(
        params_init=py_utils.WeightInit.Gaussian(scale=0.01),
        input_dim=self.MODEL_DIM * 2,
        num_classes=self.NUM_LABELS,
    )

    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(
        p.decoder.softmax)
    sm_params = p.decoder.softmax.Copy()
    sm_params.input_dim = self.MODEL_DIM
    shared_emb = lingvo_layers.SharedSoftmaxLayer.Params().Set(
        softmax=sm_params, num_classes=sm_params.num_classes)
    shared_emb.scale_sqrt_depth = True
    shared_emb.softmax.use_num_classes_major_weight = False
    p.decoder.shared_emb = shared_emb
    p.encoder.shared_emb = shared_emb

    p.decoder_ex.softmax = model_helper.ChangeToSimpleSoftmax(
        p.decoder_ex.softmax)
    sm_params_ex = p.decoder_ex.softmax.Copy()
    sm_params_ex.input_dim = self.MODEL_DIM
    shared_emb_ex = lingvo_layers.SharedSoftmaxLayer.Params().Set(
        softmax=sm_params_ex, num_classes=sm_params.num_classes)
    shared_emb_ex.scale_sqrt_depth = True
    shared_emb_ex.softmax.use_num_classes_major_weight = True
    p.decoder_ex.shared_emb_ex = shared_emb_ex
    p.encoder_ex.shared_emb_ex = shared_emb_ex

    p = self._ConfigureTaskMixed(p)
    self._SetLearner(p)
    p.train.max_steps = self.MAX_STEPS

    return p

@model_registry.RegisterSingleTaskModel
class SST2ClassificationMixed(MixedFinetuneTemplate):
  """Mixed Finetune SST-2 task from two pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  """
  GLUE_TASK_NAME = 'glue_sst2_v002'
  LABEL_NAMES = ['negative', 'positive']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 67349
  NUM_EVAL_SAMPLES = 872

  # Training param
  # LEARNING_RATE = 5e-5
  # LEARNING_RATE = 3e-5
  LEARNING_RATE = 3e-5
  # LEARNING_RATE = 1e-5
  TRAIN_BATCH_SIZE = 512
  EVAL_BATCH_SIZE = 128


@model_registry.RegisterSingleTaskModel
class QNLIClassificationMixed(MixedFinetuneTemplate):
  """Finetune QNLI task from two pretrained encoder-decoder model.

  TODO(ziruiw): Add mldash link.
  """
  GLUE_TASK_NAME = 'glue_qnli_v002'
  LABEL_NAMES = ['not_entailment', 'entailment']
  NUM_LABELS = len(LABEL_NAMES)
  NUM_TRAIN_SAMPLES = 104743
  NUM_EVAL_SAMPLES = 5463
  LEARNING_RATE = 3e-5
