"""Tasks."""
import itertools
import math
from typing import Any, Dict, Text, Tuple, List

import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import hyperparams
from lingvo.core import layers as lingvo_layers
from lingvo.core import layers_with_attention
from lingvo.core import batch_major_attention
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import summary_utils as lingvo_summary_utils
from lingvo.core.ops import record_pb2
import numpy as np

from google3.learning.brain.research.babelfish import aux_metrics
from google3.learning.brain.research.babelfish import metrics as bf_metrics
from google3.learning.brain.research.babelfish import py_utils_tpu
from google3.learning.brain.research.babelfish.multimodal import dalle
from google3.learning.brain.research.babelfish.multimodal import layers
from google3.learning.brain.research.babelfish.multimodal import metrics as multimodal_metrics
from google3.learning.brain.research.babelfish.multimodal import objectives
from google3.learning.brain.research.babelfish.multimodal import summary_utils
from google3.learning.brain.research.babelfish.multimodal import tfhub_encoders
from google3.learning.brain.research.babelfish.multimodal import vqgan
from google3.learning.deepmind.tensorflow.einshape import tf_ops


class MultimodalBaseTask(base_model.BaseTask):
  """Base task for multimodal modeling.

  In addition to base_model.BaseTask, MultimodalBaseTask runs batch_processors
  before ComputePredictions and ComputeLoss.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'batch_processors', [],
        'A list of batch_processors specifying batch preprocessors to apply '
        'sequentially. Batch preprocessors are expected to run on device (TPU)')
    p.Define('moe_aux_loss_weight', 0.0,
             'Weight of the mixture-of-experts aux loss if present.')
    return p

  def __init__(self, params: hyperparams.Params):
    super().__init__(params)
    p = self.params
    assert p.name
    self.CreateChildren('batch_processors', p.batch_processors)

  def ProcessInputBatch(self, theta: py_utils.NestedMap,
                        input_batch: py_utils.NestedMap) -> py_utils.NestedMap:
    """Processes input batch based on input_batch.

    Additionally processes input batch before feeding input to the rest of the
    model.

    Args:
      theta: A NestedMap of theta.
      input_batch: A NestedMap containing input tensors in the format returned
        by input_batch.

    Returns:
      A NestedMap containing preprocessed inputs to feed to the model.
    """
    for processor in self.batch_processors:
      if not processor.supports_tpu:
        raise ValueError(
            f'Processor {processor.params.name} does not support TPU.')
      input_batch = processor.Process(input_batch, is_batch=True)
      tf.logging.info(
          f'Processor {processor.params.name} output: {input_batch!r}.')
    return input_batch

  def ComputePredictionsWithAuxLoss(
      self,
      theta: py_utils.NestedMap,
      input_batch: py_utils.NestedMap,
  ) -> py_utils.NestedMap:
    """Computes predictions with optional auxiliary loss.

    Args:
      theta: Model weights.
      input_batch: Model inputs.

    Returns:
      Predictions of model with additional key 'moe_aux_loss' if
        p.moe_aux_loss_weight != 0.
    """
    p = self.params
    if p.moe_aux_loss_weight == 0.0:
      return self.ComputePredictions(theta, input_batch)

    with layers_with_attention.AuxLossContext() as aux_loss_ctx:
      predictions = self.ComputePredictions(theta, input_batch)
      if aux_loss_ctx.aux_losses:
        aux_loss = tf.add_n(aux_loss_ctx.aux_losses)
      else:
        aux_loss = tf.constant(0.0, dtype=py_utils.FPropDtype(p))
      predictions['moe_aux_loss'] = aux_loss
    return predictions

  def ComputeLossWithAuxLoss(
      self,
      theta: py_utils.NestedMap,
      predictions: py_utils.NestedMap,
      input_batch: py_utils.NestedMap,
  ) -> Tuple[Dict[str, Tuple[tf.Tensor, tf.Tensor]], Dict[Any, Any]]:
    """Computes loss with optional auxiliary loss.

    Args:
      theta: A `py_utils.NestedMap` of theta (model weights).
      predictions: A `py_utils.NestedMap` object containing predictions of this
        task.
      input_batch: A `py_utils.NestedMap` object containing input tensors to
        this tower.

    Returns:
      (dict, dict):

      - A dict containing str keys and (metric, weight) pairs as values, where
        one of the keys is expected to be 'loss'.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    p = self.params
    ret_metrics, ret_per_seq = self.ComputeLoss(theta, predictions, input_batch)
    if p.moe_aux_loss_weight == 0.0:
      return ret_metrics, ret_per_seq
    moe_aux_loss = p.moe_aux_loss_weight * predictions.moe_aux_loss
    loss, num_valid_examples = ret_metrics['loss']
    ret_metrics['loss'] = (loss + moe_aux_loss, num_valid_examples)
    ret_metrics['moe_aux_loss'] = (predictions.moe_aux_loss, num_valid_examples)
    ret_metrics.update(py_utils.GetTpuSummaryTensors())
    return ret_metrics, ret_per_seq

  def FPropTower(
      self,
      theta: py_utils.NestedMap,
      input_batch: py_utils.NestedMap,
  ) -> Tuple[Dict[str, Tuple[tf.Tensor, tf.Tensor]], Dict[Any, Any]]:
    """Forward propagation through one tower of the model.

    Args:
      theta: A `py_utils.NestedMap` object containing variable values of this
        task copied to this tower's devices.
      input_batch: A `py_utils.NestedMap` object containing input tensors to
        this tower.

    Returns:
      (dict, dict):

      - A dict containing str keys and (metric, weight) pairs as values, where
        one of the keys is expected to be 'loss'.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    input_batch = self.ProcessInputBatch(theta, input_batch)
    predictions = self.ComputePredictionsWithAuxLoss(theta, input_batch)
    return self.ComputeLossWithAuxLoss(theta, predictions, input_batch)


class ClassificationTask(MultimodalBaseTask):
  """Image classification task."""

  @classmethod
  def Params(cls) -> hyperparams.Params:
    p = super().Params()
    p.Define('softmax', lingvo_layers.SimpleFullSoftmax.Params(),
             'Softmax layer.')
    p.Define('network', None,
             'Params of the network to extract image features.')
    p.Define('input_fields', ['image'],
             'List of fields in input_batch to pass as inputs to the network.')
    p.Define(
        'num_average_preds', 0,
        'The number of predictions to average together during eval. We '
        'assume the examples are arranged in <(bn), ...> (batch first).')
    p.Define(
        'multilabel', False, 'If true use multilabel metrics (i.e. remove '
        'accuracy, add mAP, mAUC, etc.).')
    return p

  def __init__(self, params: hyperparams.Params):
    super().__init__(params)
    p = self.params
    assert p.name
    self.CreateChild('network', p.network)
    self.CreateChild('softmax', p.softmax)

  def ComputePredictions(self, theta, input_batch):
    p = self.params
    inputs = [input_batch.Get(k) for k in self.params.input_fields]
    features = self.network.FProp(theta.network, *inputs)
    logits = self.softmax.Logits(theta.softmax, features)
    probs = tf.nn.softmax(logits, axis=-1)
    if self.do_eval and p.num_average_preds > 1:
      # Return the probs of averaged views if p.num_average_preds > 1.
      probs = tf_ops.einshape('(bn)...->bn...', probs, n=p.num_average_preds)
      probs = tf.reduce_mean(probs, axis=1, keepdims=False)
      # Return the logits and features of first view if p.num_average_preds > 1.
      logits = logits[::p.num_average_preds, ...]
      features = features[::p.num_average_preds, ...]
    return py_utils.NestedMap(features=features, logits=logits, probs=probs)

  def ComputeLoss(
      self, theta: py_utils.NestedMap, predictions: py_utils.NestedMap,
      input_batch: py_utils.NestedMap
  ) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any]]:
    p = self.params
    batch_size = py_utils.GetShape(input_batch.label_probs)[0]
    example_weights = tf.ones(batch_size)
    if 'weight' in input_batch:
      example_weights = input_batch.weight
      example_weights = py_utils.HasShape(example_weights, [batch_size])
    num_valid_examples = tf.reduce_sum(example_weights)
    per_example_xent, _ = self.softmax.XentLossFromLogits(
        theta=theta.softmax,
        logits=predictions.logits,
        class_weights=example_weights,
        class_probabilities=input_batch.label_probs)
    avg_xent = tf.reduce_sum(per_example_xent * example_weights) / tf.maximum(
        1.0, num_valid_examples)
    rets = {
        'loss': (avg_xent, num_valid_examples),
        'log_pplx': (avg_xent, num_valid_examples),
        'num_preds': (num_valid_examples, 1),
    }
    if self.do_eval and not p.multilabel:
      acc1 = objectives.top_k_accuracy(
          1,
          predictions.probs,
          label_probs=input_batch.label_probs,
          weights=example_weights)
      acc5 = objectives.top_k_accuracy(
          5,
          predictions.probs,
          label_probs=input_batch.label_probs,
          weights=example_weights)
      rets.update(
          accuracy=(acc1, num_valid_examples),
          acc5=(acc5, num_valid_examples),
          error=(1.0 - acc1, num_valid_examples),
          error5=(1.0 - acc5, num_valid_examples))
    return rets, {'loss': per_example_xent}

  def Inference(self) -> Dict[Text, Any]:
    """Constructs the inference subgraphs.

    Returns:
      {'subgraph_name': (fetches, feeds)}
    """
    subgraphs = {}
    with tf.name_scope('inference'):
      with tf.name_scope('feature'):
        subgraphs['recordstr_feature'] = self._Inference_RecordStr_Feature()
        subgraphs['image_bytes_feature'] = self._Inference_ImageBytes_Feature()
    return subgraphs

  def _Inference_RecordStr_Feature(self) -> Any:
    """Inference support to extract feature vectors.

    Returns:
      (fetches, feeds)
    """
    feeds, fetches = py_utils.NestedMap(), py_utils.NestedMap()
    # The infeeded string can be a serialized TFRecord, AnnotatedImages, etc.,
    # depending on the input generator used here.
    feeds.recordstr = tf.placeholder(tf.string, shape=[1])
    batch = self.input_generator.ParseTFRecords(feeds.recordstr)
    fetches.embedding = self.network.FProp(self.theta.network, batch.image)
    return fetches, feeds

  def _Inference_ImageBytes_Feature(self) -> Any:
    """Inference support to extract feature vectors from image bytes.

    Returns:
      (fetches, feeds)
    """
    feeds, fetches = py_utils.NestedMap(), py_utils.NestedMap()
    # The infeeded string is a jpeg or png, etc.
    feeds.image_bytes = tf.placeholder(tf.string, shape=[])
    batch = self.input_generator.ImageBytesToBatch(feeds.image_bytes)
    fetches.embedding = self.network.FProp(self.theta.network, batch.image)
    # A scalar
    fetches.embedding = fetches.embedding[0]
    return fetches, feeds

  def CreateDecoderMetrics(self) -> Dict[Text, Any]:
    """Creates a dict of decoder metrics for `PostProcessDecodeOut` to update.

    Returns:
      A dict mapping from string keys to `.BaseMetric` objects.
    """
    p = self.params
    output = {'num_samples_in_batch': metrics.AverageMetric()}

    if p.multilabel:
      num_classes = p.softmax.num_classes
      output['auc_pr'] = metrics.MultiClassAUCMetric(num_classes, mode='pr')
      output['auc_roc'] = metrics.MultiClassAUCMetric(num_classes, mode='roc')
    else:
      output['accuracy'] = metrics.AverageMetric()
      output['acc5'] = metrics.AverageMetric()
    return output

  def DecodeWithTheta(self, theta: py_utils.NestedMap,
                      input_batch: py_utils.NestedMap) -> py_utils.NestedMap:
    """Constructs the decode graph for decoding with theta."""
    p = self.params
    input_batch = self.ProcessInputBatch(theta, input_batch)
    ret = self.ComputePredictions(theta, input_batch)
    if not p.multilabel:
      label_ids = tf.math.argmax(input_batch.label_probs, axis=-1)
      ret.correct_top1 = tf.nn.in_top_k(
          targets=label_ids, predictions=ret.probs, k=1)
      ret.correct_top5 = tf.nn.in_top_k(
          targets=label_ids, predictions=ret.probs, k=5)
    if p.multilabel:
      ret.label_probs = input_batch.label_probs
      ret.pred_probs = tf.math.sigmoid(ret.logits)

    if 'image_ids' in input_batch:
      ret.image_ids = input_batch.image_ids
    batch_size = py_utils.GetShape(input_batch.label_probs)[0]
    if 'weight' in input_batch:
      ret.weight = input_batch.weight
    else:
      ret.weight = tf.ones(batch_size)
    return ret

  def SerializeOutputs(self, nmap):
    """Return a serialized representation of the contents of `nmap`.

    Args:
      nmap: A NestedMap of data to serialize.

    Returns:
      A serialized record_pb2.Record() of the contents of `nmap`.
    """
    record = record_pb2.Record()
    flat_nmap = nmap.FlattenItems()
    for key, value in flat_nmap:
      record.fields[key].CopyFrom(tf.make_tensor_proto(value))
    serialized = record.SerializeToString()
    return serialized

  def MultiLabelDecodeOut(self, batch_size: int, decode_out_dict: Dict[str,
                                                                       Any],
                          decode_metrics_dict: Dict[str, Any]) -> List[Any]:
    p = self.params
    batch_size = decode_out_dict['weight'].shape[0]
    outputs = []

    # Move from [b, classes] numpy array -> list of lists sliced along axis=1.
    label_class_list, prob_class_list = [], []
    for i in range(p.softmax.num_classes):
      label_class_list.append([
          val[0] for val in np.take(
              decode_out_dict['label_probs'], indices=[i], axis=1).tolist()
      ])
      prob_class_list.append([
          val[0] for val in np.take(
              decode_out_dict['pred_probs'], indices=[i], axis=1).tolist()
      ])
    decode_metrics_dict['auc_pr'].Update(
        labels=label_class_list,
        probs=prob_class_list,
        weights=decode_out_dict['weight'].tolist())
    decode_metrics_dict['auc_roc'].Update(
        labels=label_class_list,
        probs=prob_class_list,
        weights=decode_out_dict['weight'].tolist())
    return outputs

  def SingleLabelDecodeOut(self, batch_size: int, decode_out_dict: Dict[str,
                                                                        Any],
                           decode_metrics_dict: Dict[str, Any]) -> List[Any]:
    outputs = []
    for batch_idx in range(batch_size):
      if decode_out_dict['weight'][batch_idx]:
        decode_metrics_dict['accuracy'].Update(
            decode_out_dict['correct_top1'][batch_idx], 1.0)
        decode_metrics_dict['acc5'].Update(
            decode_out_dict['correct_top5'][batch_idx], 1.0)
    return outputs

  def PostProcessDecodeOut(self, decode_out_dict: Dict[str, Any],
                           decode_metrics_dict: Dict[str, Any]) -> List[Any]:
    """Post-processes decoder out and updates contents of `decode_metrics_dict`.

    Args:
      decode_out_dict: A dictionary of Tensors fetched.
      decode_metrics_dict: A dict mapping from string key to `.BaseMetric`
        object as created by `CreateDecoderMetrics`.

    Returns:
      output_key_value_pairs - a list of (key, value) pairs that can be saved
      (i.e. of type str, bytes, or unicode).
    """
    p = self.params
    batch_size = decode_out_dict['weight'].shape[0]
    decode_metrics_dict['num_samples_in_batch'].Update(batch_size)

    if p.multilabel:
      outputs = self.MultiLabelDecodeOut(batch_size, decode_out_dict,
                                         decode_metrics_dict)
    else:
      outputs = self.SingleLabelDecodeOut(batch_size, decode_out_dict,
                                          decode_metrics_dict)
    for batch_idx in range(batch_size):
      if (decode_out_dict['weight'][batch_idx] and
          'image_ids' in decode_out_dict):
        image_id = decode_out_dict['image_ids'][batch_idx]
        example_output = py_utils.NestedMap(
            logits=decode_out_dict['logits'][batch_idx])
        outputs.append((image_id, self.SerializeOutputs(example_output)))
    return outputs


class SimCLRTask(ClassificationTask):
  """SimCLR representation learning task.

  A Simple Framework for Contrastive Learning of Visual Representations:
  https://arxiv.org/abs/2002.05709
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'projector', layers.ProjectionMLP.Params(),
        'Projector MLP used on top of the network for unsupervised learning.')
    p.Define('num_views', 2,
             'An integer as the number of views for contrastive learning.')
    p.Define(
        'contrastive_loss',
        objectives.ContrastiveLossLayer.Params().Set(
            intraview_contrast=True, temperature=0.1),
        'A configurable contrastive loss layer for contrastive learning.')
    p.Define(
        'loss_from_view_id_pairs', None,
        'A list of integer tuples (i, j) indicating that the total loss will '
        'include the representation loss between view i and view j. '
        'For example, [(0, 1), (2, 4)] indicates that only the loss between '
        'view pair 0 and 1 and pair 2 and 4 will be considered. If not set, '
        'by default we compute losses from combinations of all views ids.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name

    # Set a list of view id pairs to compute average loss.
    if p.loss_from_view_id_pairs is None:
      # By default set to combination of all view ids.
      p.loss_from_view_id_pairs = itertools.combinations(range(p.num_views), 2)

    if p.projector is not None:
      self.CreateChild('projector', p.projector)

    self.CreateChild('contrastive_loss', p.contrastive_loss)

  def ComputePredictions(self, theta, input_batch):
    """Computes predictions from input views.

    Args:
      theta: A NestedMap of theta.
      input_batch: A NestedMap of input.

    Returns:
      A NestedMap whose values are lists of length num_views (V) with the
    following:
      - views: `V` Tensors of [B, H, W, C] floating images.
      - embeddings: `V` Tensors of [B, K] floating point embedding vectors.
      - projections: `V` Tensors of [B, P] floating point projection vectors.
    """
    p = self.params
    # get views
    views = [input_batch.GetItem('view_%d' % i) for i in range(p.num_views)]
    embeddings = [self.network.FProp(theta.network, view) for view in views]

    # Compute network projections with output shape: [B, C].
    projections = [
        self.projector.FProp(theta.projector, embedding)
        for embedding in embeddings
    ]

    return py_utils.NestedMap({
        'views': views,
        'embeddings': embeddings,
        'projections': projections,
    })

  def _ComputeRepresentationLearningLoss(self, theta, predictions, input_batch):
    ret_metrics = {}
    ret_per_seq = {}
    batch_size = py_utils.GetShape(input_batch.weight)[0]
    repr_losses = []
    for id_pairs, (proj_x, proj_y) in zip(
        itertools.combinations(range(self.params.num_views), 2),
        itertools.combinations(predictions.projections, 2)):
      if id_pairs in self.params.loss_from_view_id_pairs:
        loss = self.contrastive_loss.FProp(theta.contrastive_loss, proj_x,
                                           proj_y)
        repr_losses.append(loss)

    # Average over losses of all pairs.
    loss = tf.math.add_n(repr_losses) / len(repr_losses)
    ret_per_seq['loss'] = loss
    ret_metrics['loss'] = (tf.reduce_mean(loss), batch_size)
    return ret_metrics, ret_per_seq

  def ComputeLoss(self, theta, predictions, input_batch):
    # Classification loss (gradient flows stopped) provides representation
    # evaluation for pretraining.
    predictions.features = tf.stop_gradient(predictions.embeddings[0])
    predictions.logits = self.softmax.Logits(theta.softmax,
                                             predictions.features)
    classifier_metrics, classifier_per_seq = super().ComputeLoss(
        theta, predictions, input_batch)

    # Unsupervised representation learning loss.
    repr_metrics, repr_per_seq = self._ComputeRepresentationLearningLoss(
        theta, predictions, input_batch)

    batch_size = py_utils.GetShape(input_batch.weight)[0]
    ret_metrics = {
        'classification_loss':
            classifier_metrics['loss'],
        'representation_learning_loss':
            repr_metrics['loss'],
        'loss': (repr_metrics['loss'][0] + classifier_metrics['loss'][0],
                 batch_size),
    }
    ret_per_seq = {
        'classification_loss': classifier_per_seq['loss'],
        'representation_learning_loss': repr_per_seq['loss'],
        'loss': classifier_per_seq['loss'] + repr_per_seq['loss'],
    }
    return ret_metrics, ret_per_seq

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the inference graph for eval decoding with theta."""
    ret = super().DecodeWithTheta(theta, input_batch)
    ret.projected = self.projector.FProp(theta.projector, ret.features)
    return ret


class BYOLTask(SimCLRTask):
  """BYOL representation learning task.

  Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning:
  https://arxiv.org/abs/2006.07733

  On top of the SimCLRTask, BYOLTask has following changes: 1) Additional
  target network which has identical architecture of p.network and p.projector
  (referred as online network in BYOL). The weights of target network are moving
  averaged values of the weights of online network; 2) Additional predictor that
  takes the input of projection of the online network, and predicts the
  projection of the target network.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'predictor', layers.ProjectionMLP.Params(),
        'Predictor MLP used for online network to predict the latent (after '
        'projector) of target network.')
    p.Define(
        'ema_update_rate_schedule', schedule.CosineSchedule.Params(),
        'EMA update rate schedule for updating moving average of variables '
        'from online network to target network. The ema_update_rate is '
        'independent of p.train.ema_decay and one can enable both.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    # Create ema_update_rate_schedule.
    self.CreateChild('ema_update_rate_schedule', p.ema_update_rate_schedule)

    # Duplicate network and projector to online and target networks.
    self.CreateChild('target_network',
                     p.network.Set(name='target_' + p.network.name))
    self.CreateChild('target_projector',
                     p.projector.Set(name='target_' + p.projector.name))

    # Create predictor of online network.
    self.CreateChild('predictor', p.predictor)

    # Save ema update variable pairs.
    self._ema_var_pairs = []

  def ComputePredictions(self, theta, input_batch):
    # Get predictions of online network from base class.
    predictions = super().ComputePredictions(theta, input_batch)

    # Compute online network predictions with output shape: [B, ..., C].
    online_predictions = [
        self.predictor.FProp(theta.predictor, proj)
        for proj in predictions.projections
    ]

    target_embeddings = [
        self.target_network.FProp(theta.target_network, view)
        for view in predictions.views
    ]

    # Compute target network projections with output shape: [B, ..., C].
    target_projections = [
        self.target_projector.FProp(theta.target_projector, target_embedding)
        for target_embedding in target_embeddings
    ]

    # Stop gradients of target projections.
    target_projections = [tf.stop_gradient(t) for t in target_projections]

    predictions.update({
        'online_predictions': online_predictions,
        'target_embeddings': target_embeddings,
        'target_projections': target_projections,
        'ema_update_rate': self.ema_update_rate_schedule.Value(),
    })
    return predictions

  def _ComputeRepresentationLearningLoss(self, theta, predictions, input_batch):
    batch_size = py_utils.GetShape(input_batch.weight)[0]
    ret_metrics = {}
    ret_per_seq = {}
    repr_losses = []
    for id_pairs, (pred_x, pred_y), (target_x, target_y) in zip(
        itertools.combinations(range(self.params.num_views), 2),
        itertools.combinations(predictions.online_predictions, 2),
        itertools.combinations(predictions.target_projections, 2)):
      if id_pairs in self.params.loss_from_view_id_pairs:
        repr_losses.append(
            objectives.compute_bootstrap_latent_loss(pred_x, pred_y, target_x,
                                                     target_y))

    # Average over losses of all pairs.
    loss = tf.math.add_n(repr_losses) / len(repr_losses)
    ret_per_seq['loss'] = loss
    ret_metrics['loss'] = (tf.reduce_mean(loss), batch_size)
    return ret_metrics, ret_per_seq

  def _GetEMAVarPairs(self):
    """Gets EMA variable pairs."""
    if self._ema_var_pairs:
      return self._ema_var_pairs

    # Get pairs of vars in network and projector.
    for online_var, target_var in zip(
        self.network.vars.Flatten() + self.projector.vars.Flatten(),
        self.target_network.vars.Flatten() +
        self.target_projector.vars.Flatten()):
      self._ema_var_pairs.append((online_var, target_var))
      tf.logging.info(
          f'Adds EMA variable pair online {online_var}, target {target_var}.')

    return self._ema_var_pairs

  def PostTrainingStepUpdate(self):
    """As a post training step, we update target network by EMA.

    target_var += (1 - p.ema_decay) * (online_var - target_var)

    At global step 1 we force the copy of weights from online network to target.

    Returns:
       Same as vars_gradients, but with dependency of assigning importance
       scores.
    """
    ema_update_rate = self.ema_update_rate_schedule.Value()

    # NOTE(jiahuiyu): The online and target network are initialized with
    # different weights, which seems fine.

    with tf.control_dependencies([
        py_utils.assert_greater_equal(ema_update_rate, tf.constant(0.)),
        py_utils.assert_less_equal(ema_update_rate, tf.constant(1.)),
    ]):
      assign_ops = []
      for online_var, target_var in self._GetEMAVarPairs():
        target_var = tf.assign_add(target_var,
                                   ema_update_rate * (online_var - target_var))
        assign_ops.append(target_var)
      return tf.group(*assign_ops)


class MERCYTask(BYOLTask):
  """Masked Encoding Regressive Consistency for unsupervised learning.

  The MERCY task aims at learning representation with minimum domain-specific
  augmentations (i.e., no color jitter, blurring, etc.). The task is based on
  BYOL by masked input encoding and regression of unmasked target encoding.

  MERCY calculates normalized regression loss between masked encoding and target
  encoding:
  - view * mask -> online network -> projector -> predictor -> masked encoding
  - view -> target network -> projector -> target encoding
  """

  def ComputePredictions(self, theta, input_batch):
    # Get a single view and masks from data.
    single_view = input_batch.view_1
    masks = input_batch.mask

    # Embeddings for supervised learning.
    embeddings = [self.network.FProp(theta.network, single_view)]

    # Construct several online views by masking the original image.
    online_views = [single_view * mask for mask in masks]

    online_embeddings = [
        self.network.FProp(theta.network, view) for view in online_views
    ]

    online_projections = [
        self.projector.FProp(theta.projector, embedding)
        for embedding in online_embeddings
    ]

    online_predictions = [
        self.predictor.FProp(theta.predictor, proj)
        for proj in online_projections
    ]

    # Compute target encoding from the original view.
    target_embeddings = [
        self.target_network.FProp(theta.target_network, single_view)
    ]

    target_projections = [
        self.target_projector.FProp(theta.target_projector, target_embedding)
        for target_embedding in target_embeddings
    ]

    # Stop gradients of target projections.
    target_projections = [tf.stop_gradient(t) for t in target_projections]

    return py_utils.NestedMap({
        'single_view': single_view,
        'embeddings': embeddings,  # for supervised loss calculation
        'online_views': online_views,
        'online_embeddings': online_embeddings,
        'online_projections': online_projections,
        'online_predictions': online_predictions,
        'target_embeddings': target_embeddings,
        'target_projections': target_projections,
        'ema_update_rate': self.ema_update_rate_schedule.Value(),
    })

  def _ComputeRepresentationLearningLoss(self, theta, predictions, input_batch):
    ret_metrics = {}
    ret_per_seq = {}
    batch_size = py_utils.GetShape(input_batch.weight)[0]
    repr_losses = []
    for pred in predictions.online_predictions:
      repr_losses.append(
          objectives.compute_normalized_regression_loss(
              pred, predictions.target_projections[0]))
    loss = tf.math.add_n(repr_losses) / len(repr_losses)
    ret_per_seq['loss'] = loss
    ret_metrics['loss'] = (tf.reduce_sum(loss), batch_size)
    return ret_metrics, ret_per_seq


class BEiTTask(ClassificationTask):
  """BEiT representation learning task.

  BEiT: BERT Pre-Training of Image Transformers
  https://arxiv.org/abs/2106.08254
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'img_quantizer', dalle.dvae_model_params_conv(8192),
        'Image quantizer on input images as visual codebooks. It should '
        'generate same grid size as image patches of vision transformer.')
    p.Define('classifier_softmax', lingvo_layers.EinsumSoftmax.Params(),
             'Classification softmax head to monitor pretraining progress.')
    p.softmax = lingvo_layers.EinsumSoftmax.Params()
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    self.CreateChild('img_quantizer', p.img_quantizer)
    self.CreateChild('classifier_softmax', p.classifier_softmax)

  def ComputePredictions(self, theta, input_batch):
    """Computes predictions from input images and masks."""
    batch_size, height, width, channels = py_utils.GetShape(input_batch.image)
    assert channels == 3
    assert height == width

    # Tokens are of shape [b, h / reduction factor, w / reduction factor].
    img_tokens = self.img_quantizer.EncodeImg(theta.img_quantizer,
                                              input_batch.image_quantizer_input)
    img_tokens = tf.stop_gradient(img_tokens)
    _, grid_height, grid_width = py_utils.GetShape(img_tokens, 3)
    img_tokens = py_utils.HasShape(img_tokens,
                                   [batch_size, grid_height, grid_width])
    flatten_img_tokens = tf.reshape(img_tokens,
                                    [batch_size, grid_height * grid_width])
    onehot_target_ids = tf.one_hot(flatten_img_tokens,
                                   self.params.softmax.num_classes)

    # Deep copy and override input_batch.label_probs and mask images.
    input_batch = tf.nest.map_structure(lambda x: x, input_batch)
    input_batch.label_probs = onehot_target_ids  # [b, seq_len, vocab_size]

    # Masked pixels are zeros and unmasked pixels are ones.
    image_mask = py_utils.HasShape(input_batch.mask, [batch_size, -1, -1, 3])
    image_mask = tf.image.resize(image_mask, [height, width], method='nearest')
    input_batch.image = input_batch.image * image_mask

    out_nmap = super().ComputePredictions(theta, input_batch)

    out_nmap['onehot_target_tokens'] = onehot_target_ids
    out_nmap['mask'] = image_mask
    out_nmap['masked_image'] = input_batch.image
    if self.do_eval:
      reconstructed_image = self.img_quantizer.DecodeImg(
          theta.img_quantizer, img_tokens)
      reconstructed_image = tf.image.resize(
          reconstructed_image, (height, width), method='nearest')
      # TODO(jiahuiyu): Need a cleaner way to deal with image ranges.
      out_nmap['reconstructed_image'] = (reconstructed_image - 0.5) / 0.5
      # TODO(jiahuiyu): Add predicted image (replacing codes with predicted in
      # masked regions).
    return out_nmap

  def ComputeLoss(self, theta, predictions, input_batch):
    loss_weight = tf.constant(1.0, dtype=tf.float32)

    # Features has shape [batch size, seq length, model dim].
    features = py_utils.HasRank(predictions.features, 3)
    # Train classifier head only to monitor pretraining progress.
    # TODO(jiahuiyu): use cls token doesn't seem to make sense here. Without
    # deep finetuning (back-prop) to backbone, the cls token likely has garbage
    # embedding?
    cls_token = tf.stop_gradient(features[:, 0, :])
    cls_logits = self.classifier_softmax.Logits(theta.classifier_softmax,
                                                cls_token)
    classifier_predictions = py_utils.NestedMap(
        features=cls_token,
        logits=cls_logits,
        probs=tf.nn.softmax(cls_logits, axis=-1))
    classifier_metrics, classifier_per_seq = super().ComputeLoss(
        theta, classifier_predictions, input_batch)
    ret_metrics = {
        f'classifier/{k}': (v[0], loss_weight)
        for k, v in classifier_metrics.items()
    }

    # Deep copy and override input_batch.label_probs as img tokens.
    input_batch = tf.nest.map_structure(lambda x: x, input_batch)
    input_batch.label_probs = tf_ops.einshape('bnc->(bn)c',
                                              predictions.onehot_target_tokens)
    input_batch.weight = tf_ops.einshape('bhw->(bhw)',
                                         1.0 - input_batch.mask[:, :, :, 0])
    seq_tokens = tf_ops.einshape('bnc->(bn)c', features[:, 1:, :])
    seq_logits = self.softmax.Logits(theta.softmax, seq_tokens)
    predictions = py_utils.NestedMap(
        features=seq_tokens,
        logits=seq_logits,
        probs=tf.nn.softmax(seq_logits, axis=-1))

    repr_metrics, repr_per_seq = super().ComputeLoss(theta, predictions,
                                                     input_batch)
    ret_metrics.update({
        f'representation_learning/{k}': (v[0], loss_weight)
        for k, v in repr_metrics.items()
    })

    ret_metrics['loss'] = (repr_metrics['loss'][0] +
                           classifier_metrics['loss'][0], loss_weight)
    ret_per_seq = {
        'classifier_loss': classifier_per_seq['loss'],
        'representation_learning_loss': repr_per_seq['loss'],
    }
    return ret_metrics, ret_per_seq

  def DecodeWithTheta(self, theta, input_batch):
    predictions = self.ComputePredictions(theta, input_batch)
    ret_nmap = py_utils.NestedMap(
        image=input_batch.image,
        masked_image=predictions.masked_image,
        reconstructed_image=predictions.reconstructed_image,
    )
    # TODO(jiahuiyu): Need a cleaner way to deal with image ranges.
    return ret_nmap.Transform(
        lambda x: tf.image.convert_image_dtype(x * 0.5 + 0.5, tf.uint8))

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    image = decode_out_dict['image']
    masked_image = decode_out_dict['masked_image']
    reconstructed_image = decode_out_dict['reconstructed_image']
    num_samples = masked_image.shape[0]
    decode_metrics_dict['num_samples_in_batch'].Update(num_samples)

    padded_imgs = summary_utils.pad_concat_images(
        [image, masked_image, reconstructed_image])
    decode_out_dict['img_summary'] = summary_utils.image_to_summary(
        padded_imgs, name='image_masked_reconstructed_predicted')
    return decode_out_dict


class ImageTextContrastiveTask(MultimodalBaseTask):
  """Image-text contrastive representation learning task.

  Ref:
  ALIGN: https://arxiv.org/abs/2102.05918
  CLIP: https://arxiv.org/abs/2103.00020
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('image_projector', layers.ProjectionMLP.Params(),
             'Projector MLP used on top of the network for image embedding.')
    p.Define('text_projector', layers.ProjectionMLP.Params(),
             'Projector MLP used on top of the network for text embedding.')
    p.Define('image_network', None,
             'Network used for producing image representations.')
    p.Define('text_network', None,
             'Network used for producing text representations.')
    # TODO(adai): Merge this into text_network.
    p.Define('text_embed_ln',
             lingvo_layers.LayerNorm.Params().Set(name='text_embed_ln',),
             'Network for text embedding layer norm.')
    p.Define(
        'contrastive_loss',
        objectives.ContrastiveLossLayer.Params().Set(
            temperature=0.1,
            learnable_temperature=True,
            l2_normalize_embedding=False,
        ), 'A configurable contrastive loss layer for contrastive learning.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    self.CreateChild('image_network', p.image_network)
    self.CreateChild('text_network', p.text_network)
    self.CreateChild('image_projector', p.image_projector)
    self.CreateChild('text_projector', p.text_projector)
    p.text_embed_ln.input_dim = p.text_projector.hidden_layer_dims[-1]
    self.CreateChild('text_embed_ln', p.text_embed_ln)
    self.CreateChild('contrastive_loss', p.contrastive_loss)

  def ComputePredictions(self, theta, input_batch):
    """Computes predictions from image-text inputs.

    Args:
      theta: A NestedMap of theta.
      input_batch: A NestedMap of input.

    Returns:
      A NestedMap whose values are lists of length 2 with the following
        image-text representations:
      - image_embedding: Tensor of [B, K] floating point embedding vectors.
      - text_embedding: Tensor of [B, N, P] floating point embedding vectors.
      - image_projection: Tensor of [B, K] floating point projection vectors.
      - text_projection: Tensor of [B, P] floating point projection vectors,
        where N is the number of text annoatations per image.
    """
    image_outputs = self._EmbedAndProjectImage(theta, input_batch)
    text_outputs = self._EmbedAndProjectText(theta, input_batch)
    return py_utils.NestedMap(**image_outputs, **text_outputs)

  def _EmbedAndProjectText(self, theta, input_batch):
    """Computes embeddings from text inputs.

    Args:
      theta: A NestedMap of theta.
      input_batch: A NestedMap of input.

    Returns:
      A NestedMap containing:
      - text_embedding: Tensor of [B, N, P] floating point embedding vectors.
      - text_projection: Tensor of [B, P] floating point projection vectors,
        where N is the number of text annoatations per image.
    """

    ids = input_batch.ids
    labels_ids = input_batch.labels
    paddings = input_batch.paddings

    batch_size, num_text_per_image, _ = py_utils.GetShape(ids)
    ids = tf.reshape(ids, [batch_size * num_text_per_image, -1])
    labels_ids = tf.reshape(labels_ids, [batch_size * num_text_per_image, -1])
    paddings = tf.reshape(paddings, [batch_size * num_text_per_image, -1])
    ids = labels_ids

    xent_out, _ = self.text_network.FProp(theta.text_network, ids, paddings)
    last_hidden = xent_out.last_hidden
    lengths = py_utils.LengthsFromPaddings(paddings)
    lengths = tf.math.maximum(lengths - 1, 0)
    text_embedding = tf.gather(last_hidden, lengths, axis=1, batch_dims=1)

    text_embedding = tf.reshape(text_embedding,
                                [batch_size, num_text_per_image, -1])
    # Compute network projections with output shape: [B, C].
    text_embedding = self.text_embed_ln.FProp(theta.text_embed_ln,
                                              text_embedding)
    text_projection = self.text_projector.FProp(theta.text_projector,
                                                text_embedding)
    text_projection = tf.math.l2_normalize(text_projection, axis=-1)

    return py_utils.NestedMap({
        'text_embedding': text_embedding,
        'text_projection': text_projection
    })

  def _EmbedAndProjectImage(self, theta, input_batch):
    """Computes embeddings from image inputs.

    Args:
      theta: A NestedMap of theta.
      input_batch: A NestedMap of input.

    Returns:
      A NestedMap containing:
      - image_embedding: Tensor of [B, K] floating point embedding vectors.
      - image_projection: Tensor of [B, K] floating point projection vectors.
    """
    image_embedding = self.image_network.FProp(theta.image_network,
                                               input_batch.image)
    image_projection = self.image_projector.FProp(theta.image_projector,
                                                  image_embedding)
    image_projection = tf.math.l2_normalize(image_projection, axis=-1)

    return py_utils.NestedMap({
        'image_embedding': image_embedding,
        'image_projection': image_projection
    })

  def _ComputeRepresentationLearningLoss(self, theta, predictions, input_batch):
    ret_metrics = {}
    ret_per_seq = {}
    batch_size = py_utils.GetShape(input_batch.weight)[0]

    # Currently we assume there's only one text per image for training.
    text_projection = py_utils.HasShape(predictions.text_projection,
                                        [batch_size, 1, -1])
    text_projection = tf.squeeze(text_projection, axis=1)
    loss = self.contrastive_loss.FProp(theta.contrastive_loss,
                                       predictions.image_projection,
                                       text_projection) / 2.0

    # Average over losses of all pairs.
    ret_per_seq['loss'] = loss
    ret_metrics['loss'] = (tf.reduce_mean(loss), batch_size)
    return ret_metrics, ret_per_seq

  def ComputeLoss(self, theta, predictions, input_batch):
    # Classification loss (gradient flows stopped) provides representation
    # evaluation for pretraining.

    # Unsupervised representation learning loss.
    repr_metrics, repr_per_seq = self._ComputeRepresentationLearningLoss(
        theta, predictions, input_batch)

    txt_proj = predictions.text_projection  # [B, 1, D]
    txt_proj = tf.squeeze(txt_proj, axis=1)  # [B, D]
    img_proj = predictions.image_projection  # [B, D]
    alignment_scores = tf.reduce_sum(img_proj * txt_proj, -1)

    batch_size = py_utils.GetShape(input_batch.weight)[0]
    ret_metrics = {
        'representation_learning_loss': repr_metrics['loss'],
        'loss': (repr_metrics['loss'][0], batch_size),
        'alignment_scores': (tf.reduce_mean(alignment_scores), batch_size),
    }
    ret_per_seq = {
        'representation_learning_loss': repr_per_seq['loss'],
        'loss': repr_per_seq['loss'],
    }
    return ret_metrics, ret_per_seq

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the inference graph for eval decoding with theta."""
    input_batch = self.ProcessInputBatch(theta, input_batch)
    ret = self.ComputePredictions(theta, input_batch)
    del ret.image_embedding
    del ret.text_embedding
    ret.weight = input_batch.weight
    ret.label_weights = input_batch.label_weights
    ret.paddings = input_batch.paddings
    return ret

  def Inference(self) -> Dict[Text, Any]:
    """Constructs the inference subgraphs.

    Returns:
      {'subgraph_name': (fetches, feeds)}
    """
    subgraphs = {}
    with tf.name_scope('inference'):
      with tf.name_scope('feature'):
        subgraphs['recordstr_feature'] = self._Inference_RecordStr_Feature()
      with tf.name_scope('image_feature'):
        subgraphs['image_feature'] = self._Inference_Image_Feature()
      with tf.name_scope('text_feature'):
        subgraphs['text_feature'] = self._Inference_Text_Feature()
    return subgraphs

  def _Inference_RecordStr_Feature(self) -> Any:
    """Inference support to extract feature vectors.

    Returns:
      (fetches, feeds)
    """
    feeds, fetches = py_utils.NestedMap(), py_utils.NestedMap()
    feeds.recordstr = tf.placeholder(tf.string, shape=[1])
    input_batch = self.input_generator.ParseTFRecords(feeds.recordstr)
    fetches = self.ComputePredictions(self.theta, input_batch)
    return fetches, feeds

  def _Inference_Image_Feature(self) -> Any:
    """Inference support to extract image embeddings from image bytes.

    Returns:
      (fetches, feeds)
    """
    feeds = py_utils.NestedMap()
    feeds.image_bytes = tf.placeholder(tf.string, shape=[])
    input_batch = self.input_generator.PreprocessImage(feeds.image_bytes)
    input_batch = input_batch.Transform(lambda x: x[tf.newaxis, ...])
    fetches = self._EmbedAndProjectImage(self.theta, input_batch)
    fetches = fetches.Transform(lambda x: x[0])
    return fetches, feeds

  def _Inference_Text_Feature(self) -> Any:
    """Inference support to extract image embeddings from text.

    Returns:
      (fetches, feeds)
    """
    feeds = py_utils.NestedMap()
    feeds.text = tf.placeholder(tf.string, shape=[])
    fed_text = tf.reshape(feeds.text, [1])
    input_batch = self.input_generator.PreprocessText(fed_text)
    input_batch = input_batch.Transform(lambda x: x[tf.newaxis, ...])
    fetches = self._EmbedAndProjectText(self.theta, input_batch)
    fetches = fetches.Transform(lambda x: x[0])
    return fetches, feeds

  def CreateDecoderMetrics(self):
    """Creates a dict of decoder metrics for `PostProcessDecodeOut` to update.

    Returns:
      A dict mapping from string keys to `.BaseMetric` objects.
    """
    return {
        'num_samples_in_batch':
            metrics.AverageMetric(),
        'retrieval':
            multimodal_metrics.RetrievalRecallMetrics.Params().Instantiate()
    }

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    """Post-processes decoder out and updates contents of `decode_metrics_dict`.

    Args:
      decode_out_dict: A dictionary of Tensors fetched.
      decode_metrics_dict: A dict mapping from string key to `.BaseMetric`
        object as created by `CreateDecoderMetrics`.

    Returns:
      output_key_value_pairs - a list of (key, value) pairs that can be saved
      (i.e. of type str, bytes, or unicode).
    """
    batch_size, _, _ = decode_out_dict['text_projection'].shape
    for batch_idx in range(batch_size):
      if decode_out_dict['weight'][batch_idx]:
        image_embedding = decode_out_dict['image_projection'][batch_idx]
        text_embedding = decode_out_dict['text_projection'][batch_idx]
        valid_text_mask = decode_out_dict['label_weights'][batch_idx] > 0
        text_embedding = text_embedding[valid_text_mask, :]
        decode_metrics_dict['retrieval'].Update(image_embedding, text_embedding)
    decode_metrics_dict['num_samples_in_batch'].Update(batch_size)
    return []


class StarburstTask(ClassificationTask):
  """Prototype task for testing co-training via go/bf-nf-distributed."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'consistency_loss_weight', 0.0,
        'Weight on remote embedding consistency loss.'
        'Remote embeddings are only used if set > 0.0')
    return p

  def _ComputeEmbedConsistencyLoss(self, theta, local, remote):
    p = self.params
    return p.consistency_loss_weight * tf.nn.l2_loss(local - remote)

  def ComputeLoss(self, theta, predictions, input_batch):
    p = self.params
    batch_size = py_utils.GetShape(input_batch.weight)[0]
    class_metrics, class_per_seq = super().ComputeLoss(theta, predictions,
                                                       input_batch)

    local_embedding = predictions.features
    local_embedding = py_utils.HasRank(local_embedding, 2)
    if p.consistency_loss_weight > 0.0:
      remote_embedding = input_batch.remote_embedding
      remote_embedding = py_utils.HasRank(remote_embedding, 2)
      consist_loss = self._ComputeEmbedConsistencyLoss(theta, local_embedding,
                                                       remote_embedding)
    else:
      consist_loss = 0.0

    ret_metrics = {
        'classification_loss': class_metrics['loss'],
        'embed_consistency_loss': (consist_loss, batch_size),
        'loss': (class_metrics['loss'][0] + consist_loss, batch_size),
    }
    return ret_metrics, class_per_seq

  def Inference(self):
    subgraphs = super().Inference()
    with tf.name_scope('inference'):
      subgraphs['serving_default'] = self._InferenceSubgraph_Starburst_Tpu()
    return subgraphs

  def _InferenceSubgraph_Starburst_Tpu(self):
    """Inference subgraph for SavedModel export."""
    images = tf.placeholder(tf.float32, shape=[None, 289, 289, 3])
    feeds = py_utils.NestedMap({'images': images})

    def _InferenceFn(images):
      return self.network.FProp(self.theta.network, images)

    rets = py_utils_tpu.RewriteForMultiCore(
        _InferenceFn, core_ordinal_feed=-1, inputs=[images])

    features = rets[0]

    fetches = py_utils.NestedMap({'embedding': features})
    return fetches, feeds


class Image2TextLMTask(MultimodalBaseTask):
  """Image to text language model with encoder-decoder architecture."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'image_embedding', None,
        'Image embedding component as entry layers for the vision modality.')
    p.Define('target_language', 'EN',
             'Target language for BLEU score calculation, default to English.')
    p.Define(
        'decode_with_ref', True,
        'If reference is available for decode. Skip interval eval if False.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    self.CreateChild('image_embedding', p.image_embedding)
    self.CreateChild('encoder', p.encoder)
    self.CreateChild('decoder', p.decoder)

  def _ComputeImageEmbedding(self, theta, input_batch):
    # image embedding
    image_encoded = self.image_embedding.FProp(theta.image_embedding,
                                               input_batch.image)
    encoded_padding = tf.zeros(
        [tf.shape(image_encoded)[0],
         tf.shape(image_encoded)[1]],
        dtype=image_encoded.dtype)

    # encoder
    sources = py_utils.NestedMap(
        input_embs=tf.transpose(image_encoded, [1, 0, 2]),
        paddings=encoded_padding)
    return self.encoder.FPropTransformerLayers(theta.encoder, sources)

  def ComputePredictions(self, theta, input_batch):
    ids = input_batch.ids
    paddings = input_batch.paddings

    batch_size, num_text_per_image, _ = py_utils.GetShape(ids)
    ids = tf.reshape(ids, [batch_size * num_text_per_image, -1])
    paddings = tf.reshape(paddings, [batch_size * num_text_per_image, -1])

    # image encoded
    encoder_outputs = self._ComputeImageEmbedding(theta, input_batch)

    # decoder
    targets = py_utils.NestedMap(ids=ids, paddings=paddings)
    predictions = self.decoder.ComputePredictions(theta.decoder,
                                                  encoder_outputs, targets)
    return predictions

  def ComputeLoss(self, theta, predictions, input_batch):
    labels_ids = input_batch.labels
    paddings = input_batch.paddings

    batch_size, num_text_per_image, _ = py_utils.GetShape(labels_ids)
    labels_ids = tf.reshape(labels_ids, [batch_size * num_text_per_image, -1])
    paddings = tf.reshape(paddings, [batch_size * num_text_per_image, -1])
    weights = 1.0 - paddings

    targets = py_utils.NestedMap(
        labels=labels_ids, weights=weights, paddings=paddings)
    return self.decoder.ComputeLoss(theta.decoder, predictions, targets)

  def CreateDecoderMetrics(self):
    p = self.params
    # BLEU score calculated for internal evaluation only and is not the same as
    # standard image captioning scores.
    all_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
    }
    if p.decode_with_ref:
      all_metrics['corpus_bleu'] = metrics.CorpusBleuMetric()
      all_metrics['canonical_bleu'] = bf_metrics.ConfigurableBleuMetric(
          p.target_language, ignore_source=True)
    return all_metrics

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the decode graph for decoding with theta."""
    encoder_outputs = self._ComputeImageEmbedding(theta, input_batch)
    encoder_outputs = self.decoder.AddExtraDecodingInfo(encoder_outputs,
                                                        input_batch)
    decoded = self.decoder.BeamSearchDecode(encoder_outputs)

    return py_utils.RunOnTpuHost(self._ProcessBeamSearchDecodeOut, input_batch,
                                 decoded)

  def _ProcessReferences(self, input_batch):
    ids = input_batch.ids
    batch_size, num_text_per_image, _ = py_utils.GetShape(ids)
    ids = tf.reshape(ids, [batch_size * num_text_per_image, -1])
    with tf.name_scope('spm_tgt'):
      targets = self.input_generator.tokenizer.IdsToStrings(ids)
    return tf.identity(targets, name='targets')

  def _ProcessBeamSearchDecodeOut(self, input_batch, decoded):
    p = self.params
    decode_outs = py_utils.NestedMap()
    batch_size = py_utils.GetShape(input_batch.image)[0]

    with tf.name_scope('spm_hyp'):
      topk_decoded = self.input_generator.tokenizer.IdsToStrings(
          decoded.topk_ids)
      topk_decoded = tf.reshape(topk_decoded, [batch_size, -1])

    if p.decode_with_ref:
      decode_outs.targets = self._ProcessReferences(input_batch)

    decode_outs.topk_decoded = tf.identity(topk_decoded, name='topk_decoded')
    if 'weight' in input_batch:
      decode_outs.weight = input_batch.weight
    else:
      decode_outs.weight = tf.ones(batch_size)
    if 'image_source_id' in input_batch:
      decode_outs.image_source_id = input_batch.image_source_id
    return decode_outs

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    """Post-processes decoder out and updates contents of `decode_metrics_dict`.

    Args:
      decode_out_dict: A dictionary of Tensors fetched.
      decode_metrics_dict: A dict mapping from string key to `.BaseMetric`
        object as created by `CreateDecoderMetrics`.

    Returns:
      outputs - a list of (key, value) pairs that can be saved
      (i.e. of type str, bytes, or unicode).
    """
    dummy_src_str = 'x'
    batch_size = decode_out_dict['weight'].shape[0]
    decode_metrics_dict['num_samples_in_batch'].Update(batch_size)

    outputs = []
    for batch_idx in range(batch_size):
      if decode_out_dict['weight'][batch_idx]:
        # Only aggregate scores of the top hypothesis.
        hyp_str = decode_out_dict['topk_decoded'][batch_idx][0]
        output_info = [hyp_str]
        if 'targets' in decode_out_dict:
          tgt_str = decode_out_dict['targets'][batch_idx]
          output_info.append(tgt_str)
          decode_metrics_dict['corpus_bleu'].Update(tgt_str, hyp_str)
          decode_metrics_dict['canonical_bleu'].Update(tgt_str, hyp_str,
                                                       dummy_src_str)
        if 'image_source_id' in decode_out_dict:
          image_id = decode_out_dict['image_source_id'][batch_idx]
          outputs.append((image_id, '%s' % output_info))
    return outputs


class Multi30kTranslation(Image2TextLMTask):
  """Image translation task with encoder-decoder architecture."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('train_mode', 'IT2T', 'Training mode, must in [T2T, IT2T]')
    p.Define('decode_mode', 'IT2T', 'Decoding mode, must in [I2T, IT2T]')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.train_mode not in ['T2T', 'IT2T']:
      raise ValueError('Unsupported train mode: {0}'.format(p.train_mode))
    if p.decode_mode not in ['I2T', 'IT2T']:
      raise ValueError('Unsupported decode mode: {0}'.format(p.decode_mode))

  # TODO(ziruiw): refactor codebase such that Image2TextLMTask is a subclass of
  # ImageText2TextLMTask, and move _ComputeXXXEmbedding functions to the base
  # class.
  def _ComputeImageTextEmbedding(self, theta, input_batch):
    # image embedding
    image_encoded = self.image_embedding.FProp(theta.image_embedding,
                                               input_batch.image)
    image_encoded = tf.transpose(image_encoded, [1, 0, 2])
    image_paddings = tf.zeros(
        [tf.shape(image_encoded)[0],
         tf.shape(image_encoded)[1]],
        dtype=image_encoded.dtype)

    # text embedding
    sources = py_utils.NestedMap(
        ids=input_batch.source_ids, paddings=input_batch.source_paddings)
    text_embeddings = self.encoder.FPropEmbeddings(theta.encoder, sources)
    text_encoded = text_embeddings.input_embs
    text_paddings = text_embeddings.paddings

    # [T, B, C]
    concat_encoded = tf.concat([image_encoded, text_encoded], axis=0)
    # [T, B]
    concat_paddings = tf.concat([image_paddings, text_paddings], axis=0)

    encoder_outputs = self.encoder.FPropTransformerLayers(
        theta.encoder,
        py_utils.NestedMap(input_embs=concat_encoded, paddings=concat_paddings))
    return encoder_outputs

  def _ComputeTextEmbedding(self, theta, input_batch):
    # text embedding
    sources = py_utils.NestedMap(
        ids=input_batch.source_ids, paddings=input_batch.source_paddings)
    text_embeddings = self.encoder.FPropEmbeddings(theta.encoder, sources)
    encoder_outputs = self.encoder.FPropTransformerLayers(
        theta.encoder, text_embeddings)
    return encoder_outputs

  def ComputePredictions(self, theta, input_batch):
    p = self.params

    # encoder
    if p.train_mode == 'T2T':
      encoder_outputs = self._ComputeTextEmbedding(theta, input_batch)
    else:
      assert p.train_mode == 'IT2T'
      encoder_outputs = self._ComputeImageTextEmbedding(theta, input_batch)

    # decoder
    targets = py_utils.NestedMap(
        ids=input_batch.target_ids, paddings=input_batch.target_paddings)
    predictions = self.decoder.ComputePredictions(theta.decoder,
                                                  encoder_outputs, targets)
    return predictions

  def ComputeLoss(self, theta, predictions, input_batch):
    labels_ids = input_batch.labels
    paddings = input_batch.target_paddings
    weights = 1.0 - paddings

    targets = py_utils.NestedMap(
        labels=labels_ids, weights=weights, paddings=paddings)
    return self.decoder.ComputeLoss(theta.decoder, predictions, targets)

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the decode graph for decoding with theta."""
    p = self.params

    # encoder
    if p.decode_mode == 'I2T':
      encoder_outputs = self._ComputeImageEmbedding(theta, input_batch)
    else:
      assert p.decode_mode == 'IT2T'
      encoder_outputs = self._ComputeImageTextEmbedding(theta, input_batch)
    encoder_outputs = self.decoder.AddExtraDecodingInfo(encoder_outputs,
                                                        input_batch)
    decoded = self.decoder.BeamSearchDecode(encoder_outputs)
    return py_utils.RunOnTpuHost(self._ProcessBeamSearchDecodeOut, input_batch,
                                 decoded)

  def _ProcessReferences(self, input_batch):
    ids = input_batch.target_ids
    with tf.name_scope('spm_tgt'):
      targets = self.input_generator.tokenizer.IdsToStrings(ids)
    return tf.identity(targets, name='targets')


class Text2TextLMTask(MultimodalBaseTask):
  """Text to text language model with encoder-decoder architecture."""

  @classmethod
  def Params(cls):
    p = super().Params()
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    self.CreateChild('encoder', p.encoder)
    self.CreateChild('decoder', p.decoder)

  def ComputePredictions(self, theta, input_batch):

    encoder_inputs = input_batch.encoder_inputs
    decoder_inputs = input_batch.decoder_inputs
    encoder_paddings = input_batch.encoder_paddings
    decoder_paddings = input_batch.decoder_paddings

    # encoder
    sources = py_utils.NestedMap(ids=encoder_inputs, paddings=encoder_paddings)
    encoder_embeddings = self.encoder.FPropEmbeddings(theta.encoder, sources)
    encoder_outputs = self.encoder.FPropTransformerLayers(
        theta.encoder, encoder_embeddings)

    # decoder
    targets = py_utils.NestedMap(ids=decoder_inputs, paddings=decoder_paddings)
    return self.decoder.ComputePredictions(theta.decoder, encoder_outputs,
                                           targets)

  def ComputeLoss(self, theta, predictions, input_batch):
    labels_ids = input_batch.labels
    paddings = input_batch.decoder_paddings

    weights = 1.0 - paddings
    labels = py_utils.NestedMap(
        labels=labels_ids, weights=weights, paddings=paddings)
    return self.decoder.ComputeLoss(theta.decoder, predictions, labels)

  def CreateDecoderMetrics(self):
    all_metrics = {
        'num_samples_in_batch':
            metrics.AverageMetric(),
        'rouge_1':
            aux_metrics.RougeMetricV2(rouge_type='rouge1', use_stemmer=True),
        'rouge_2':
            aux_metrics.RougeMetricV2(rouge_type='rouge2', use_stemmer=True),
        'rouge_L':
            aux_metrics.RougeMetricV2(rouge_type='rougeL', use_stemmer=True),
    }
    return all_metrics

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the decode graph for decoding with theta."""
    encoder_inputs = input_batch.encoder_inputs
    encoder_paddings = input_batch.encoder_paddings

    # encoder
    sources = py_utils.NestedMap(ids=encoder_inputs, paddings=encoder_paddings)
    encoder_embeddings = self.encoder.FPropEmbeddings(theta.encoder, sources)
    encoder_outputs = self.encoder.FPropTransformerLayers(
        theta.encoder, encoder_embeddings)
    encoder_outputs = self.decoder.AddExtraDecodingInfo(encoder_outputs,
                                                        input_batch)
    decoded = self.decoder.BeamSearchDecode(encoder_outputs)

    return py_utils.RunOnTpuHost(self._ProcessBeamSearchDecodeOut, input_batch,
                                 decoded)

  def _ProcessBeamSearchDecodeOut(self, input_batch, decoded):
    decode_outs = py_utils.NestedMap()
    ids = input_batch.decoder_inputs
    batch_size = py_utils.GetShape(ids)[0]

    with tf.name_scope('spm_hyp'):
      topk_decoded = self.input_generator.tokenizer.IdsToStrings(
          decoded.topk_ids)
      topk_decoded = tf.reshape(topk_decoded, [batch_size, -1])

    with tf.name_scope('spm_tgt'):
      targets = self.input_generator.tokenizer.IdsToStrings(ids)
      decode_outs.targets = tf.identity(targets, name='targets')

    decode_outs.topk_decoded = tf.identity(topk_decoded, name='topk_decoded')
    if 'weight' in input_batch:
      decode_outs.weight = input_batch.weight
    else:
      decode_outs.weight = tf.ones([batch_size])
    return decode_outs

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    """Post-processes decoder out and updates contents of `decode_metrics_dict`.

    Args:
      decode_out_dict: A dictionary of Tensors fetched.
      decode_metrics_dict: A dict mapping from string key to `.BaseMetric`
        object as created by `CreateDecoderMetrics`.

    Returns:
      outputs - a list of (key, value) pairs that can be saved
      (i.e. of type str, bytes, or unicode).
    """
    batch_size = decode_out_dict['weight'].shape[0]
    decode_metrics_dict['num_samples_in_batch'].Update(batch_size)

    outputs = []
    for batch_idx in range(batch_size):
      if decode_out_dict['weight'][batch_idx]:
        # Only aggregate scores of the top hypothesis.
        hyp_str = decode_out_dict['topk_decoded'][batch_idx][0]
        if 'targets' in decode_out_dict:
          tgt_str = decode_out_dict['targets'][batch_idx]
          decode_metrics_dict['rouge_1'].Update(tgt_str, hyp_str)
          decode_metrics_dict['rouge_2'].Update(tgt_str, hyp_str)
          decode_metrics_dict['rouge_L'].Update(tgt_str, hyp_str)
          outputs.append((hyp_str, tgt_str))
    return outputs


class ImageText2TextLMTask(Image2TextLMTask):
  """Image/Text to text language model with encoder-decoder architecture."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('mdl_type', 'enc_dec',
             ('Mdl type of this model, must be in ["dec_only", "enc_dec"].'
              'dec_only uses a single prefix lm decoder where the image is used'
              'as the prefix.'))
    return p

  def __init__(self, params):
    # TODO(ziruiw): enable decoder-only option for parent class and remove this.
    # pylint: disable=bad-super-call
    super(Image2TextLMTask, self).__init__(params)
    # pylint: enable=bad-super-call
    p = self.params
    assert p.name
    if p.mdl_type not in ['enc_dec', 'dec_only']:
      raise ValueError('Unsupported model type: {0}'.format(p.mdl_type))
    assert p.image_embedding
    assert p.decoder
    self.CreateChild('image_embedding', p.image_embedding)
    self.CreateChild('decoder', p.decoder)

    if p.mdl_type == 'enc_dec':
      assert p.encoder
      self.CreateChild('encoder', p.encoder)

  def ComputeText2TextPredictions(self, theta, input_batch):
    p = self.params
    if p.mdl_type == 'dec_only':
      return self.decoder.ComputePredictions(theta.decoder, None, input_batch)
    elif p.mdl_type == 'enc_dec':
      encoder_inputs = input_batch.encoder_inputs
      decoder_inputs = input_batch.decoder_inputs
      encoder_paddings = input_batch.encoder_paddings
      decoder_paddings = input_batch.decoder_paddings

      # encoder
      sources = py_utils.NestedMap(
          ids=encoder_inputs, paddings=encoder_paddings)
      encoder_embeddings = self.encoder.FPropEmbeddings(theta.encoder, sources)
      encoder_outputs = self.encoder.FPropTransformerLayers(
          theta.encoder, encoder_embeddings)

      # decoder
      targets = py_utils.NestedMap(
          ids=decoder_inputs, paddings=decoder_paddings)
      return self.decoder.ComputePredictions(theta.decoder, encoder_outputs,
                                             targets)

  def ComputeImage2TextPredictions(self, theta, input_batch):
    p = self.params
    text_ids = input_batch.ids
    text_paddings = input_batch.paddings

    batch_size, num_text_per_image, _ = py_utils.GetShape(text_ids)
    text_ids = tf.reshape(text_ids, [batch_size * num_text_per_image, -1])
    text_paddings = tf.reshape(text_paddings,
                               [batch_size * num_text_per_image, -1])

    # image embedding
    image_encoded = self.image_embedding.FProp(theta.image_embedding,
                                               input_batch.image)

    image_paddings = tf.zeros(
        [tf.shape(image_encoded)[0],
         tf.shape(image_encoded)[1]],
        dtype=image_encoded.dtype)

    if p.mdl_type == 'enc_dec':
      # encoder
      sources = py_utils.NestedMap(
          input_embs=tf.transpose(image_encoded, [1, 0, 2]),
          paddings=image_paddings)
      encoder_outputs = self.encoder.FPropTransformerLayers(
          theta.encoder, sources)

      # decoder
      targets = py_utils.NestedMap(ids=text_ids, paddings=text_paddings)
      return self.decoder.ComputePredictions(theta.decoder, encoder_outputs,
                                             targets)

    elif p.mdl_type == 'dec_only':
      # text embedding
      text_inputs = py_utils.NestedMap(ids=text_ids, paddings=text_paddings)
      text_encoded = self.decoder.FPropEmbeddings(theta.decoder, text_inputs)

      # joint embedding
      # [batch, total_len, dim]
      transformer_input = tf.concat(
          [image_encoded, text_encoded.transformer_input], axis=1)
      paddings = tf.concat([image_paddings, text_encoded.paddings], axis=1)

      # Compute visibility where images can only attend to images and text can
      # attend to both image and previous text.
      image_len = py_utils.GetShape(image_encoded)[1]
      text_len = py_utils.GetShape(text_ids)[1]
      image_visibility = tf.zeros([1, image_len], dtype=tf.int32)
      text_visibility = tf.reshape(tf.range(text_len), [1, text_len]) + 1
      token_visibility = tf.tile(
          tf.concat([image_visibility, text_visibility], axis=1),
          [batch_size, 1])

      decoder_inputs = py_utils.NestedMap(
          transformer_input=transformer_input,
          paddings=paddings,
          token_visibility=token_visibility)

      predictions = self.decoder.FPropTransformerLayers(theta.decoder, None,
                                                        decoder_inputs)
      predictions.token_visibility = token_visibility
      return predictions

  def ComputeImage2TextLMLoss(self, theta, predictions, input_batch):
    labels_ids = input_batch.labels
    paddings = input_batch.paddings

    batch_size, num_text_per_image, _ = py_utils.GetShape(labels_ids)
    labels_ids = tf.reshape(labels_ids, [batch_size * num_text_per_image, -1])
    paddings = tf.reshape(paddings, [batch_size * num_text_per_image, -1])
    weights = 1.0 - paddings

    targets = py_utils.NestedMap(
        labels=labels_ids, weights=weights, paddings=paddings)
    return self.decoder.ComputeLoss(theta.decoder, predictions, targets)

  def ComputeText2TextLMLoss(self, theta, predictions, input_batch):
    p = self.params
    if p.mdl_type == 'dec_only':
      # Only include useful fields to avoid error during batch-major transpose.
      labels = input_batch.labels
      paddings = input_batch.paddings
      weights = input_batch.weights
    elif p.mdl_type == 'enc_dec':
      labels = input_batch.labels
      paddings = input_batch.decoder_paddings
      weights = 1.0 - paddings
    targets = py_utils.NestedMap(
        labels=labels, weights=weights, paddings=paddings)
    return self.decoder.ComputeLoss(theta.decoder, predictions, targets)

  def _CombineMultiTaskMetrics(self, task_metrics):
    """Combine the Metrics from multi-task input generators."""
    combined_metrics = {'loss': (0.0, 0.0)}
    task_weights = self.params.input.task_weights if self.params.input else {
        'Image2TextLM': 1.,
        'Text2TextLM': 1.
    }

    for task_name, sub_task_metrics in task_metrics.items():
      task_weight = task_weights[task_name]
      combined_metrics.update({
          f'{task_name}/{metric_name}': value
          for metric_name, value in sub_task_metrics.items()
      })
      combined_metrics['loss'] = (combined_metrics['loss'][0] +
                                  sub_task_metrics['loss'][0] * task_weight,
                                  combined_metrics['loss'][1] +
                                  sub_task_metrics['loss'][1] * task_weight)
    return combined_metrics

  def ComputePredictions(self, theta, input_batch):
    predictions = py_utils.NestedMap()
    predictions['Image2TextLM'] = self.ComputeImage2TextPredictions(
        theta, input_batch['Image2TextLM'])
    predictions['Text2TextLM'] = self.ComputeText2TextPredictions(
        theta, input_batch['Text2TextLM'])
    return predictions

  def ComputeLoss(self, theta, predictions, input_batch):
    task_metrics = py_utils.NestedMap()
    task_metrics['Image2TextLM'] = self.ComputeImage2TextLMLoss(
        theta, predictions['Image2TextLM'], input_batch['Image2TextLM'])[0]
    task_metrics['Text2TextLM'] = self.ComputeText2TextLMLoss(
        theta, predictions['Text2TextLM'], input_batch['Text2TextLM'])[0]
    return self._CombineMultiTaskMetrics(task_metrics), {}


class VQAClassification(MultimodalBaseTask):
  """VQA classification task."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_classes', 3129, 'Number of classes.')
    p.Define('token_type_emb', None, 'Embedding layer params.')
    p.Define(
        'image_embedding', None,
        'Image embedding component as entry layers for the vision modality.')
    p.Define('projector', lingvo_layers.FCLayer.Params(), 'Logits output.')
    # TODO(ziruiw): consolidate into a single projector with LN support.
    p.Define('projector_ln', lingvo_layers.LayerNorm.Params(),
             'Optional layer norm of MLP.')
    p.Define(
        'softmax', layers.SimpleFullSigmoidCrossEntropy.Params(),
        'Softmax layer for VQA. We are using standard binary Xent loss,'
        'following piror work.')
    p.Define('mdl_type', 'enc_only',
             'Mdl type of this model, must be in ["enc_only", "enc_dec"]')
    p.Define(
        'pool_type', 'last',
        'Pool type for classification, must be in ["max", "mean", "last"]')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    if p.mdl_type not in ['enc_only', 'enc_dec']:
      raise ValueError('Unsupported model type: {0}'.format(p.mdl_type))
    if p.pool_type not in ['max', 'mean', 'last']:
      raise ValueError('Unsupported pooling type: {0}'.format(p.pool_type))
    if p.token_type_emb:
      self.CreateChild('token_type_emb', p.token_type_emb)
    if p.image_embedding:
      self.CreateChild('image_embedding', p.image_embedding)
    self.CreateChild('encoder', p.encoder)
    self.CreateChild('projector', p.projector)

    if p.mdl_type in ['enc_dec']:
      self.CreateChild('decoder', p.decoder)

    if p.projector_ln:
      ln_params = p.projector_ln.Copy()
      ln_params.name = 'projector_ln'
      ln_params.input_dim = p.projector.output_dim
      self.CreateChild('projector_ln', ln_params)

    self.CreateChild('softmax', p.softmax)

  def _apply_classifier(self, theta, classifier_input):
    p = self.params
    # logits
    features = self.projector.FProp(theta.projector, classifier_input)
    if p.projector_ln:
      features = self.projector_ln.FProp(theta.projector_ln, features)

    logits = self.softmax.Logits(theta.softmax, features)
    probs = tf.nn.softmax(logits, axis=-1)
    return py_utils.NestedMap(logits=logits, probs=probs, features=features)

  def _extract_classifier_input(self, paddings, decoder_outputs):
    """Compute classifier input from decoder output based on pool_type.

    Args:
      paddings: A tensor of paddings of decoder input of shape [B, T].
      decoder_outputs: A tensor of decoder output of shape [T, B, C] or [B, T,
        C], depending on the output format of decoder.

    Returns:
      classifier_input: A tensor of input to classifier network of shape [B, C].
    """

    p = self.params
    # Transpose decoder_output to shape [B, T, C] if needed.
    if ('prediction_data_format' not in p.decoder or
        p.decoder.prediction_data_format == 'TBC'):
      decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])

    batch_size, length = py_utils.GetShape(paddings)
    decoder_outputs = py_utils.HasShape(decoder_outputs,
                                        [batch_size, length, -1])

    # TODO(ziruiw): experiment with using embedding of special token, e.g. EOS
    # [B]
    seq_len = tf.reduce_sum(1. - paddings, axis=1)
    if p.pool_type == 'mean':
      # average pooling of the text
      # [B, T, C]
      softmax_input = py_utils.ApplyPadding(
          tf.expand_dims(paddings, -1), decoder_outputs)
      # [B, C]
      softmax_input = tf.reduce_sum(softmax_input, axis=1)
      classifier_input = tf.einsum('b,bc->bc', 1. / seq_len, softmax_input)
    elif p.pool_type == 'max':
      # max pooling of the text
      # [B, T, C]
      softmax_input = py_utils.ApplyPadding(
          tf.expand_dims(paddings, -1),
          decoder_outputs,
          padded=tf.ones_like(decoder_outputs) * tf.float32.min)
      # [B, C]
      classifier_input = tf.reduce_max(softmax_input, axis=1)
    else:
      assert p.pool_type == 'last'
      # last token of the text
      # [B, T]
      last_token_one_hot = tf.one_hot(
          tf.cast(seq_len, tf.int32) - 1,
          py_utils.GetShape(paddings)[1],
          dtype=decoder_outputs.dtype)
      # [B, C]
      classifier_input = tf.einsum('bt,btc->bc', last_token_one_hot,
                                   decoder_outputs)
    return classifier_input

  def ComputePredictions(self, theta, input_batch):
    p = self.params

    ids = input_batch.ids
    paddings = input_batch.paddings

    # image embedding
    # [Batch, Length, Dim]
    image_embedding = self.image_embedding.FProp(theta.image_embedding,
                                                 input_batch.image)
    image_paddings = tf.zeros(
        [tf.shape(image_embedding)[0],
         tf.shape(image_embedding)[1]],
        dtype=image_embedding.dtype)
    # TODO(ziruiw): switch to batch_major and remove this line.
    image_embedding = tf.transpose(image_embedding, [1, 0, 2])

    if p.mdl_type == 'enc_only':
      # text embedding
      text_inputs = py_utils.NestedMap(ids=ids, paddings=paddings)
      text_encoded = self.encoder.FPropEmbeddings(theta.encoder, text_inputs)
      text_embedding, text_paddings = (text_encoded.input_embs,
                                       text_encoded.paddings)

      # apply token type embedding for image and text modality
      if p.token_type_emb:
        image_type = tf.zeros(
            [
                tf.shape(image_embedding)[0] - 1,  # to be concat with CLS token
                tf.shape(image_embedding)[1]
            ],
            dtype=ids.dtype)
        image_type_embedding = self.token_type_emb.EmbLookup(
            theta.token_type_emb, tf.reshape(image_type, [-1]))
        image_type_embedding = tf.reshape(image_type_embedding, [
            tf.shape(image_embedding)[0] - 1, -1, p.token_type_emb.embedding_dim
        ])

        cls_type_embedding = tf.zeros([
            1,
            tf.shape(image_type_embedding)[1],
            tf.shape(image_type_embedding)[2]
        ],
                                      dtype=image_type_embedding.dtype)
        image_embedding += tf.concat([cls_type_embedding, image_type_embedding],
                                     axis=0)

        text_type = tf.ones(
            [tf.shape(text_embedding)[0],
             tf.shape(text_embedding)[1]],
            dtype=ids.dtype)
        text_type_embedding = self.token_type_emb.EmbLookup(
            theta.token_type_emb, tf.reshape(text_type, [-1]))
        text_type_embedding = tf.reshape(
            text_type_embedding,
            [tf.shape(text_embedding)[0], -1, p.token_type_emb.embedding_dim])
        text_embedding += text_type_embedding

      # image + text embedding
      input_embedding = tf.concat([image_embedding, text_embedding], axis=0)
      input_paddings = tf.concat([image_paddings, text_paddings], axis=1)

      # encoder
      inputs = py_utils.NestedMap(
          input_embs=input_embedding, paddings=input_paddings)
      encoder_outputs = self.encoder.FPropTransformerLayers(
          theta.encoder, inputs)
      classifier_input = encoder_outputs.encoded[0, :, :]

    elif p.mdl_type == 'enc_dec':
      # encoder
      sources = py_utils.NestedMap(
          input_embs=image_embedding, paddings=image_paddings)
      encoder_outputs = self.encoder.FPropTransformerLayers(
          theta.encoder, sources)

      # decoder
      targets = py_utils.NestedMap(ids=ids, paddings=paddings)
      decoder_outputs = self.decoder.ComputePredictions(theta.decoder,
                                                        encoder_outputs,
                                                        targets)
      if isinstance(decoder_outputs, py_utils.NestedMap):
        decoder_outputs = decoder_outputs.softmax_input
      classifier_input = self._extract_classifier_input(paddings,
                                                        decoder_outputs)
    return self._apply_classifier(theta, classifier_input)

  def ComputeLoss(self, theta, predictions, input_batch):
    p = self.params

    labels_id = input_batch.labels
    pad_weights = input_batch.weight

    # we use class_id = num_classes as the UNK class
    one_hot_labels = tf.one_hot(labels_id, p.num_classes + 1)
    label_counts = tf.reduce_sum(one_hot_labels, axis=1)[:, :p.num_classes]

    # vqa score = min{#match with human labels/3, 1}, averaged over all
    # 10 choose 9 sets of human annotators
    scores = tf.minimum(1.0, 0.3 * label_counts)
    label_unk_weights = tf.greater(tf.reduce_sum(scores, axis=1), 0.0)
    label_unk_weights = tf.cast(label_unk_weights, pad_weights.dtype)

    example_weights = label_unk_weights * pad_weights
    num_valid_examples = tf.reduce_sum(pad_weights)
    num_train_examples = tf.reduce_sum(example_weights)

    per_example_xent, per_example_argmax = self.softmax.XentLossFromLogits(
        theta=theta.softmax,
        logits=predictions.logits,
        class_weights=None,
        class_probabilities=scores)
    per_example_xent = per_example_xent * example_weights
    avg_xent = tf.reduce_sum(per_example_xent) / tf.maximum(
        1e-8, num_train_examples)

    per_example_acc = tf.reduce_sum(
        tf.one_hot(per_example_argmax, p.num_classes) * scores, axis=1)
    acc = tf.reduce_sum(per_example_acc * pad_weights) / tf.maximum(
        1e-8, num_valid_examples)
    rets = {
        'loss': (avg_xent, num_train_examples),
        'num_train_examples': (num_train_examples, 1),
        'num_valid_examples': (num_valid_examples, 1),
        'vqa_score': (acc, num_valid_examples),
    }
    return rets, {'loss': per_example_xent}


class GLUEClassification(MultimodalBaseTask):
  """GLUE classification task.

  This task takes the same input format of T5 models with a single formatted
  string for each example in the GLUE task. It then follows the same training
  paradigm of BART where the same input is fed into both the encoder and the
  decoder, and a classifier module is added on top.

  TODO(ziruiw): refactor code to unify pipeline for t2t, i2t and it2t models.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_classes', 2, 'Number of classes.')
    p.Define('token_type_emb', None, 'Embedding layer params.')
    p.Define('projector', lingvo_layers.FCLayer.Params(), 'Logits output.')
    p.Define('projector_ln', lingvo_layers.LayerNorm.Params(),
             'Optional layer norm of MLP.')
    p.Define(
        'softmax', layers.SimpleFullSigmoidCrossEntropy.Params(),
        'Softmax layer for VQA. We are using standard binary Xent loss,'
        'following piror work.')
    p.Define('mdl_type', 'enc_dec',
             'Mdl type of this model, must be in ["enc_only", "enc_dec"]')
    p.Define(
        'pool_type', 'last',
        'Pool type for classification, must be in ["max", "mean", "last"]')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    if p.mdl_type not in ['enc_only', 'enc_dec']:
      raise ValueError('Unsupported model type: {0}'.format(p.mdl_type))
    if p.pool_type not in ['max', 'mean', 'last']:
      raise ValueError('Unsupported pooling type: {0}'.format(p.pool_type))
    if p.token_type_emb:
      self.CreateChild('token_type_emb', p.token_type_emb)

    self.CreateChild('encoder', p.encoder)
    self.CreateChild('projector', p.projector)

    if p.mdl_type in ['enc_dec']:
      self.CreateChild('decoder', p.decoder)

    if p.projector_ln:
      ln_params = p.projector_ln.Copy()
      ln_params.name = 'projector_ln'
      ln_params.input_dim = p.projector.output_dim
      self.CreateChild('projector_ln', ln_params)

    self.CreateChild('softmax', p.softmax)

  def _apply_classifier(self, theta, classifier_input):
    p = self.params
    # logits
    features = self.projector.FProp(theta.projector, classifier_input)
    if p.projector_ln:
      features = self.projector_ln.FProp(theta.projector_ln, features)

    logits = self.softmax.Logits(theta.softmax, features)
    probs = tf.nn.softmax(logits, axis=-1)
    return py_utils.NestedMap(logits=logits, probs=probs, features=features)

  def _extract_classifier_input(self, paddings, decoder_outputs):
    """Compute classifier input from decoder output based on pool_type.

    Args:
      paddings: A tensor of paddings of decoder input of shape [B, T].
      decoder_outputs: A tensor of decoder output of shape [T, B, C] or [B, T,
        C], depending on the output format of decoder.

    Returns:
      classifier_input: A tensor of input to classifier network of shape [B, C].
    """

    p = self.params
    # Transpose decoder_output to shape [B, T, C] if needed.
    if ('prediction_data_format' not in p.decoder or
        p.decoder.prediction_data_format == 'TBC'):
      decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])

    batch_size, length = py_utils.GetShape(paddings)
    decoder_outputs = py_utils.HasShape(decoder_outputs,
                                        [batch_size, length, -1])

    seq_len = tf.reduce_sum(1. - paddings, axis=1)
    if p.pool_type == 'mean':
      # average pooling of the text
      # [B, T, C]
      softmax_input = py_utils.ApplyPadding(
          tf.expand_dims(paddings, -1), decoder_outputs)
      # [B, C]
      softmax_input = tf.reduce_sum(softmax_input, axis=1)
      classifier_input = tf.einsum('b,bc->bc', 1. / seq_len, softmax_input)
    elif p.pool_type == 'max':
      # max pooling of the text
      # [B, T, C]
      softmax_input = py_utils.ApplyPadding(
          tf.expand_dims(paddings, -1),
          decoder_outputs,
          padded=tf.ones_like(decoder_outputs) * tf.float32.min)
      # [B, C]
      classifier_input = tf.reduce_max(softmax_input, axis=1)
    else:
      assert p.pool_type == 'last'
      # last token of the text
      # [B, T]
      last_token_one_hot = tf.one_hot(
          tf.cast(seq_len, tf.int32) - 1,
          py_utils.GetShape(paddings)[1],
          dtype=decoder_outputs.dtype)
      # [B, C]
      classifier_input = tf.einsum('bt,btc->bc', last_token_one_hot,
                                   decoder_outputs)
    return classifier_input

  def ComputePredictions(self, theta, input_batch):
    ids = input_batch.ids
    paddings = input_batch.paddings

    # encoder
    sources = py_utils.NestedMap(ids=ids, paddings=paddings)
    encoder_embeddings = self.encoder.FPropEmbeddings(theta.encoder, sources)
    encoder_outputs = self.encoder.FPropTransformerLayers(
        theta.encoder, encoder_embeddings)

    # decoder
    targets = py_utils.NestedMap(ids=ids, paddings=paddings)
    decoder_outputs = self.decoder.ComputePredictions(theta.decoder,
                                                      encoder_outputs, targets)

    classifier_input = self._extract_classifier_input(paddings, decoder_outputs)
    return self._apply_classifier(theta, classifier_input)

  def ComputeLoss(self, theta, predictions, input_batch):
    p = self.params
    batch_size = py_utils.GetShape(input_batch.labels)[0]
    example_weights = tf.ones(batch_size)
    if 'weight' in input_batch:
      example_weights = input_batch.weight
      example_weights = py_utils.HasShape(example_weights, [batch_size])
    num_valid_examples = tf.reduce_sum(example_weights)

    labels = tf.reshape(input_batch.labels, [-1])
    one_hot_labels = tf.one_hot(labels, p.num_classes)
    per_example_xent, _ = self.softmax.XentLossFromLogits(
        theta=theta.softmax,
        logits=predictions.logits,
        class_weights=example_weights,
        class_probabilities=one_hot_labels)
    avg_xent = tf.reduce_sum(per_example_xent) / tf.maximum(
        1.0, num_valid_examples)

    rets = {
        'loss': (avg_xent, num_valid_examples),
        'num_valid_examples': (num_valid_examples, 1),
    }

    acc1 = objectives.top_k_accuracy(
        1, predictions.probs, label_ids=labels, weights=example_weights)
    rets.update(
        accuracy=(acc1, num_valid_examples),
        error=(1. - acc1, num_valid_examples))
    return rets, {'loss': per_example_xent}

  def CreateDecoderMetrics(self):
    """Creates a dict of decoder metrics for `PostProcessDecodeOut` to update.

    Returns:
      A dict mapping from string keys to `.BaseMetric` objects.
    """
    all_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'accuracy': metrics.AverageMetric(),
    }
    if self.params.num_classes == 2:
      all_metrics['mcc'] = metrics.MCCMetric()
      all_metrics['f1'] = metrics.F1Metric()
    return all_metrics

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the decode graph for decoding with theta."""
    ret = self.ComputePredictions(theta, input_batch)
    batch_size = py_utils.GetShape(input_batch.labels)[0]
    if 'weight' in input_batch:
      ret.weight = input_batch.weight
    else:
      ret.weight = tf.ones(batch_size)

    label_ids = tf.reshape(input_batch.labels, [-1])
    ret.labels = label_ids
    ret.correct_top1 = tf.nn.in_top_k(
        targets=label_ids, predictions=ret.probs, k=1)
    return ret

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    """Post-processes decoder out and updates contents of `decode_metrics_dict`.

    Args:
      decode_out_dict: A dictionary of Tensors fetched.
      decode_metrics_dict: A dict mapping from string key to `.BaseMetric`
        object as created by `CreateDecoderMetrics`.

    Returns:
      outputs - a list of (key, value) pairs that can be saved
      (i.e. of type str, bytes, or unicode).
    """
    batch_size = decode_out_dict['weight'].shape[0]
    outputs = []
    for batch_idx in range(batch_size):
      if decode_out_dict['weight'][batch_idx]:
        is_correct = decode_out_dict['correct_top1'][batch_idx]
        label_id = decode_out_dict['labels'][batch_idx]
        decode_metrics_dict['accuracy'].Update(is_correct, 1.0)
        if self.params.num_classes == 2:
          if is_correct and label_id:
            decode_metrics_dict['mcc'].UpdateTruePositive()
            decode_metrics_dict['f1'].UpdateTruePositive()
          elif is_correct and not label_id:
            decode_metrics_dict['mcc'].UpdateTrueNegative()
          elif not is_correct and label_id:
            decode_metrics_dict['mcc'].UpdateFalsePositive()
            decode_metrics_dict['f1'].UpdateFalsePositive()
          else:
            decode_metrics_dict['mcc'].UpdateFalseNegative()
            decode_metrics_dict['f1'].UpdateFalseNegative()

    decode_metrics_dict['num_samples_in_batch'].Update(batch_size)
    return outputs

class MixedFinetune(MultimodalBaseTask):
  """Mixed Finetune task.

  This task fine tune the pretrained IT2T and T2T models by adaptively selecting
  the embeddings. It takes the same input format of T5 models with a single
  formatted string for each example in the GLUE task. It then follows the same
  training paradigm of BART where the same input is fed into both the encoder
  and the decoder, and a classifier module is added on top.

  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_classes', 2, 'Number of classes.')
    p.Define('token_type_emb', None, 'Embedding layer params.')

    p.Define('encoder_ex', None, 'Extra Encoder Params.')
    p.Define('decoder_ex', None, 'Extra Decoder Params.')

    p.Define('emb_remap', lingvo_layers.FCLayer.Params(),
             'Map the it2t embeddings to the t2t space')
    p.Define('emb_selector', lingvo_layers.FCLayer.Params(),
             'Concatenate two embeddings and select one')

    p.Define('projector', lingvo_layers.FCLayer.Params(),
             'Logits output.')
    p.Define('projector_ln', lingvo_layers.LayerNorm.Params(),
             'Optional layer norm of MLP.')
    p.Define(
        'softmax', layers.SimpleFullSigmoidCrossEntropy.Params(),
        'Softmax layer for VQA. We are using standard binary Xent loss,'
        'following piror work.')
    p.Define('mdl_type', 'enc_dec',
             'Mdl type of this model, must be in ["enc_only", "enc_dec"]')
    p.Define(
        'pool_type', 'last',
        'Pool type for classification, must be in ["max", "mean", "last"]')

    tp = p.train
    tp.Define(
        'init_from_multiple_checkpoint_rules', {},
        'If not None, a dictionary with keys corresponding to a checkpoint '
        'path and values corresponding to variable loading rules is expected. '
        'Each key is expected to be a path to a checkpoint from which to '
        'initialize part of the model. Variables are only loaded from this '
        'path during initialization and will override values provided by '
        'initialization.\n'
        'The corresponding values (loading_rules) are expected to be a tuple '
        'consisting of two list: loading rules, and ignore rules, respectively.'
        'The first list (loading rules) contains the list of variables '
        'which should be initialized from the checkpoint: each element in the '
        'list is a pair of strings. The first element is a regex and the '
        'second is a python format string. If a variable in the model matches '
        'a regex, we rename using the format string to determine the '
        'corresponding var in the checkpoint. If a model variable would match '
        'multiple loading rules, the first rule that matches is used.\n'
        'The second list (ignore rules) is a list of regexes which specify '
        'variables in the model which should not be initialized using the '
        'loading rules. Thus, if a variable in the model to be trained matches '
        'one of the rules in the loading rules, as well as one of the regular '
        'expressions in the ignore rules, the variable will not be initialized '
        'from the checkpoint, but will instead be initialized from the '
        'variable initalizer defined in the graph.'
        'Examples:'
        '{"checkpoint_path": ([("(.*)", "%s")], [])} will initialize all the '
        'model parameters from the checkpoint_path.')

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    if p.mdl_type not in ['enc_only', 'enc_dec']:
      raise ValueError('Unsupported model type: {0}'.format(p.mdl_type))
    if p.pool_type not in ['max', 'mean', 'last']:
      raise ValueError('Unsupported pooling type: {0}'.format(p.pool_type))
    if p.token_type_emb:
      self.CreateChild('token_type_emb', p.token_type_emb)

    self.CreateChild('encoder', p.encoder)
    self.CreateChild('encoder_ex', p.encoder_ex)
    self.CreateChild('projector', p.projector)

    if p.mdl_type in ['enc_dec']:
      self.CreateChild('decoder', p.decoder)
      self.CreateChild('decoder_ex', p.decoder_ex)

    if p.emb_selector:
      self.CreateChild('emb_remap', p.emb_remap)
      self.CreateChild('emb_selector', p.emb_selector)

    if p.projector_ln:
      ln_params = p.projector_ln.Copy()
      ln_params.name = 'projector_ln'
      ln_params.input_dim = p.projector.output_dim
      self.CreateChild('projector_ln', ln_params)

    self.CreateChild('softmax', p.softmax)

  def _apply_classifier(self, theta, classifier_input):
    p = self.params
    # logits
    features = self.projector.FProp(theta.projector, classifier_input)
    if p.projector_ln:
      features = self.projector_ln.FProp(theta.projector_ln, features)

    logits = self.softmax.Logits(theta.softmax, features)
    probs = tf.nn.softmax(logits, axis=-1)
    return py_utils.NestedMap(logits=logits, probs=probs, features=features)

  def _extract_classifier_input(self, paddings, decoder_outputs):
    """Compute classifier input from decoder output based on pool_type.

    Args:
      paddings: A tensor of paddings of decoder input of shape [B, T].
      decoder_outputs: A tensor of decoder output of shape [T, B, C] or [B, T,
        C], depending on the output format of decoder.

    Returns:
      classifier_input: A tensor of input to classifier network of shape [B, C].
    """

    p = self.params
    # Transpose decoder_output to shape [B, T, C] if needed.
    if ('prediction_data_format' not in p.decoder or
        p.decoder.prediction_data_format == 'TBC'):
      decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])

    batch_size, length = py_utils.GetShape(paddings)
    decoder_outputs = py_utils.HasShape(decoder_outputs,
                                        [batch_size, length, -1])

    seq_len = tf.reduce_sum(1. - paddings, axis=1)
    if p.pool_type == 'mean':
      # average pooling of the text
      # [B, T, C]
      softmax_input = py_utils.ApplyPadding(
          tf.expand_dims(paddings, -1), decoder_outputs)
      # [B, C]
      softmax_input = tf.reduce_sum(softmax_input, axis=1)
      classifier_input = tf.einsum('b,bc->bc', 1. / seq_len, softmax_input)
    elif p.pool_type == 'max':
      # max pooling of the text
      # [B, T, C]
      softmax_input = py_utils.ApplyPadding(
          tf.expand_dims(paddings, -1),
          decoder_outputs,
          padded=tf.ones_like(decoder_outputs) * tf.float32.min)
      # [B, C]
      classifier_input = tf.reduce_max(softmax_input, axis=1)
    else:
      assert p.pool_type == 'last'
      # last token of the text
      # [B, T]
      last_token_one_hot = tf.one_hot(
          tf.cast(seq_len, tf.int32) - 1,
          py_utils.GetShape(paddings)[1],
          dtype=decoder_outputs.dtype)
      # [B, C]
      classifier_input = tf.einsum('bt,btc->bc', last_token_one_hot,
                                   decoder_outputs)
    return classifier_input

  def EncFPropMixedEmbeddings(self, theta, input_batch):
    p = self.encoder.params
    if p.packed_input:
      raise ValueError('not supported for now')

    p = self.encoder.params
    with tf.name_scope("mixed_enc"):
      # [batch, time]
      input_ids = input_batch.ids
      # [batch, time]
      paddings = input_batch.paddings

      batch = py_utils.GetShape(input_ids)[0]
      time = py_utils.GetShape(input_ids)[1]

      # Embedding layer.
      # [batch, time, dim]
      if not p.shared_emb:
        input_embs = self.encoder.token_emb.EmbLookup(
            theta.encoder.token_emb, input_ids)
        input_embs_ex = self.encoder_ex.token_emb.EmbLookup(
            theta.encoder_ex.token_emb, input_ids)
      else:
        input_embs = self.encoder.softmax.EmbLookup(
            theta.encoder.softmax, input_ids)
        input_embs_ex = self.encoder_ex.softmax.EmbLookup(
            theta.encoder_ex.softmax, input_ids)

      # Concatenate token embeddings along dim
      selector_input = tf.concat([input_embs, input_embs_ex], axis=2)
      # Select between IT2T and T2T embeddings
      selection = self.emb_selector.FProp(theta.emb_selector, selector_input)
      # Transform IT2T embedding into the same space
      trans_input_embs_ex = self.emb_remap.FProp(theta.emb_remap, input_embs_ex)
      # Linear combination
      input_embs = selection * input_embs + (1-selection) * trans_input_embs_ex

      # [1, time, dim]
      position_embs = tf.expand_dims(
          self.encoder.position_emb.FProp(theta.encoder.position_emb, time), 0)

      # [batch, time, dim]
      input_embs += position_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)
        paddings = tf.cast(paddings, p.input_dropout_tpl.fprop_dtype)

      input_embs = self.encoder.input_dropout.FProp(theta.encoder.input_dropout, input_embs)
      # [batch, time, dim]
      transformer_input = input_embs
      # Explicitly set the input shape of Transformer layers, to avoid
      # unknown shape error occurred to tf.einsum on nonTPU devices.
      transformer_input = tf.reshape(transformer_input,
                                     [batch, time, p.model_dim])

      # Reshape to match with input shapes of other embeddings, e.g. image.
      transformer_input = tf.transpose(transformer_input, [1, 0, 2])
      paddings = tf.transpose(paddings)

    return py_utils.NestedMap(input_embs=transformer_input, paddings=paddings)

  def DecFPropMixedEmbeddings(self, theta, encoder_outputs, targets):
    p = self.decoder.params
    # [batch, source_time, dim]
    encoder_out_bm = self.decoder._MaybeTransposeEncoderOutputs(encoder_outputs, 'BTC')
    aux_vec = encoder_out_bm.encoded
    aux_paddings = encoder_out_bm.padding
    aux_segment_id = getattr(encoder_out_bm, 'segment_id', None)

    with tf.name_scope("mixed_dec"):
      # [batch, target_time]
      target_ids = targets.ids
      target_paddings = targets.paddings
      target_time = py_utils.GetShape(target_ids)[1]
      target_segment_pos = None
      target_segment_id = None
      if p.packed_input:
        target_segment_id = targets.segment_ids
        target_segment_pos = targets.segment_pos
        assert aux_segment_id is not None, ('Need to provide aux_segment_id '
                                            'for packed input.')

      # Embedding layer.
      # [batch, time, dim]
      if not p.shared_emb:
        input_embs = self.decoder.token_emb.EmbLookup(
            theta.decoder.token_emb, target_ids)
        input_embs_ex = self.decoder.token_emb.EmbLookup(
            theta.decoder_ex.token_emb, target_ids)
      else:
        input_embs = self.decoder.softmax.EmbLookup(
            theta.decoder.softmax, target_ids)
        input_embs_ex = self.decoder_ex.softmax.EmbLookup(
            theta.decoder_ex.softmax, target_ids)

      # Concatenate token embeddings along dim
      selector_input = tf.concat([input_embs, input_embs_ex], axis=2)
      # Select between IT2T and T2T embeddings
      selection = self.emb_selector.FProp(theta.emb_selector, selector_input)
      # Transform IT2T embedding into the same space
      trans_input_embs_ex = self.emb_remap.FProp(theta.emb_remap, input_embs_ex)
      # Linear combination
      input_embs = selection * input_embs + (1-selection) * trans_input_embs_ex


      # Embedding layer
      # [batch, target_time, dim]
      if not p.shared_emb:
        token_embs = self.decoder.token_emb.EmbLookup(
            theta.decoder.token_emb, target_ids)
        token_embs_ex = self.decoder_ex.token_emb.EmbLookup(
            theta.decoder_ex.token_emb, target_ids)
      else:
        token_embs = self.decoder.softmax.EmbLookup(
            theta.decoder.softmax, target_ids)
        token_embs_ex = self.decoder_ex.softmax.EmbLookup(
            theta.decoder_ex.softmax, target_ids)

      # Concatenate token embeddings along dim
      selector_input = tf.concat([token_embs, token_embs_ex], axis=2)
      # Select between IT2T and T2T embeddings
      selection = self.emb_selector.FProp(theta.emb_selector, selector_input)
      # Transform IT2T embedding into the same space
      trans_token_embs_ex = self.emb_remap.FProp(theta.emb_remap, token_embs_ex)
      # Linear combination
      token_embs = selection * token_embs + (1-selection) * trans_token_embs_ex

      # [1, target_time, dim]
      if p.packed_input:
        posit_embs = self.decoder.position_emb.FPropWithPosition(
            theta.decoder.position_emb, target_segment_pos)
      else:
        posit_embs = tf.expand_dims(
            self.decoder.position_emb.FProp(theta.decoder.position_emb, target_time), 0)
      # [batch, target_time, dim]
      input_embs = token_embs + posit_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)
        target_paddings = tf.cast(target_paddings,
                                  p.input_dropout_tpl.fprop_dtype)

      input_embs = self.decoder.input_dropout.FProp(
                        theta.decoder.input_dropout, input_embs)
      layer_in = input_embs
      # Explicitly set the input shape of Transformer layers, to avoid
      # unknown shape error occurred to tf.einsum on nonTPU devices.
      batch, _, dim = py_utils.GetShape(aux_vec, 3)
      layer_in = tf.reshape(layer_in, [batch, target_time, dim])
      if p.packed_input:
        segment_padding = batch_major_attention.SegmentMask(
            target_segment_id,
            target_segment_id,
            dtype=layer_in.dtype,
            apply_dtype_min=False)
        causal_padding = tf.expand_dims(
            tf.tile(
                tf.expand_dims(
                    batch_major_attention.CausalPadding(
                        target_time, dtype=layer_in.dtype), 0), [batch, 1, 1]),
            1)
        segment_padding = tf.math.maximum(causal_padding, segment_padding)
        segment_mask = segment_padding * batch_major_attention.GetDtypeMin(
            dtype=layer_in.dtype)
        aux_segment_mask = batch_major_attention.SegmentMask(
            target_segment_id, aux_segment_id, dtype=layer_in.dtype)
      for layer, layer_theta in zip(self.decoder.decoder_trans,
                                    theta.decoder.decoder_trans):
        # [batch, target_time, dim]
        shape = py_utils.GetShape(layer_in)
        batch_size = shape[0]
        seq_len = shape[1]
        target_paddings = tf.reshape(target_paddings, [batch_size, seq_len])
        layer_out, _ = layer.FProp(
            layer_theta,
            layer_in,
            target_paddings,
            aux_vec,
            aux_paddings,
            segment_mask=segment_mask if p.packed_input else None,
            aux_segment_mask=aux_segment_mask if p.packed_input else None)
        layer_in = layer_out

      if p.final_layer_norm:
        layer_out = self.decoder.final_ln.FProp(theta.decoder.final_ln, layer_out)
      if p.prediction_data_format == 'TBC':
        # Transpose the softmax_input to match the input requirement of
        # ComputePredictions.
        layer_out = tf.transpose(layer_out, [1, 0, 2])
      return layer_out

  def ComputePredictions(self, theta, input_batch):
    ids = input_batch.ids
    paddings = input_batch.paddings

    # encoder with embedding mixture
    sources = py_utils.NestedMap(ids=ids, paddings=paddings)

    encoder_embeddings = self.EncFPropMixedEmbeddings(theta, sources)

    encoder_outputs = self.encoder.FPropTransformerLayers(
        theta.encoder, encoder_embeddings)

    # decoder
    targets = py_utils.NestedMap(ids=ids, paddings=paddings)

    decoder_outputs = self.DecFPropMixedEmbeddings(theta,
                                                   encoder_outputs, targets)

    classifier_input = self._extract_classifier_input(paddings, decoder_outputs)
    return self._apply_classifier(theta, classifier_input)

  def ComputeLoss(self, theta, predictions, input_batch):
    p = self.params
    batch_size = py_utils.GetShape(input_batch.labels)[0]
    example_weights = tf.ones(batch_size)
    if 'weight' in input_batch:
      example_weights = input_batch.weight
      example_weights = py_utils.HasShape(example_weights, [batch_size])
    num_valid_examples = tf.reduce_sum(example_weights)

    labels = tf.reshape(input_batch.labels, [-1])
    one_hot_labels = tf.one_hot(labels, p.num_classes)
    per_example_xent, _ = self.softmax.XentLossFromLogits(
        theta=theta.softmax,
        logits=predictions.logits,
        class_weights=example_weights,
        class_probabilities=one_hot_labels)
    avg_xent = tf.reduce_sum(per_example_xent) / tf.maximum(
        1.0, num_valid_examples)

    rets = {
        'loss': (avg_xent, num_valid_examples),
        'num_valid_examples': (num_valid_examples, 1),
    }

    acc1 = objectives.top_k_accuracy(
        1, predictions.probs, label_ids=labels, weights=example_weights)
    rets.update(
        accuracy=(acc1, num_valid_examples),
        error=(1. - acc1, num_valid_examples))
    return rets, {'loss': per_example_xent}

  def CreateDecoderMetrics(self):
    """Creates a dict of decoder metrics for `PostProcessDecodeOut` to update.

    Returns:
      A dict mapping from string keys to `.BaseMetric` objects.
    """
    all_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'accuracy': metrics.AverageMetric(),
    }
    if self.params.num_classes == 2:
      all_metrics['mcc'] = metrics.MCCMetric()
      all_metrics['f1'] = metrics.F1Metric()
    return all_metrics

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the decode graph for decoding with theta."""
    ret = self.ComputePredictions(theta, input_batch)
    batch_size = py_utils.GetShape(input_batch.labels)[0]
    if 'weight' in input_batch:
      ret.weight = input_batch.weight
    else:
      ret.weight = tf.ones(batch_size)

    label_ids = tf.reshape(input_batch.labels, [-1])
    ret.labels = label_ids
    ret.correct_top1 = tf.nn.in_top_k(
        targets=label_ids, predictions=ret.probs, k=1)
    return ret

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    """Post-processes decoder out and updates contents of `decode_metrics_dict`.

    Args:
      decode_out_dict: A dictionary of Tensors fetched.
      decode_metrics_dict: A dict mapping from string key to `.BaseMetric`
        object as created by `CreateDecoderMetrics`.

    Returns:
      outputs - a list of (key, value) pairs that can be saved
      (i.e. of type str, bytes, or unicode).
    """
    batch_size = decode_out_dict['weight'].shape[0]
    outputs = []
    for batch_idx in range(batch_size):
      if decode_out_dict['weight'][batch_idx]:
        is_correct = decode_out_dict['correct_top1'][batch_idx]
        label_id = decode_out_dict['labels'][batch_idx]
        decode_metrics_dict['accuracy'].Update(is_correct, 1.0)
        if self.params.num_classes == 2:
          if is_correct and label_id:
            decode_metrics_dict['mcc'].UpdateTruePositive()
            decode_metrics_dict['f1'].UpdateTruePositive()
          elif is_correct and not label_id:
            decode_metrics_dict['mcc'].UpdateTrueNegative()
          elif not is_correct and label_id:
            decode_metrics_dict['mcc'].UpdateFalsePositive()
            decode_metrics_dict['f1'].UpdateFalsePositive()
          else:
            decode_metrics_dict['mcc'].UpdateFalseNegative()
            decode_metrics_dict['f1'].UpdateFalseNegative()

    decode_metrics_dict['num_samples_in_batch'].Update(batch_size)
    return outputs


class WNLIClassification(GLUEClassification):
  @classmethod
  def Params(cls):
    p = super().Params()
    return p

  def ComputePredictions(self, theta, input_batch):
    ids = input_batch.ids
    paddings = input_batch.paddings

    # encoder
    sources = py_utils.NestedMap(ids=ids, paddings=paddings)
    encoder_embeddings = self.encoder.FPropEmbeddings(theta.encoder, sources)
    encoder_outputs = self.encoder.FPropTransformerLayers(
        theta.encoder, encoder_embeddings)

    # decoder
    targets = py_utils.NestedMap(ids=ids, paddings=paddings)
    decoder_outputs = self.decoder.ComputePredictions(theta.decoder,
                                                      encoder_outputs, targets)

    classifier_input = self._extract_classifier_input(paddings, decoder_outputs)
    return self._apply_classifier(theta, classifier_input)


class SNLIVEClassification(GLUEClassification):
  """SNLI-VE task of classification."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('zero_shot_transfer_mode', True,
             'If true, train on text-only NLI data.')
    return p

  def ComputePredictions(self, theta, input_batch):
    p = self.params

    if self.do_eval or not p.zero_shot_transfer_mode:
      hypothesis_ids = input_batch.hypothesis_ids
      hypothesis_paddings = input_batch.hypothesis_paddings

      # image embedding
      # [Batch, Length, Dim]
      image_embedding = self.image_embedding.FProp(theta.image_embedding,
                                                   input_batch.image)
      image_paddings = tf.zeros(
          [tf.shape(image_embedding)[0],
           tf.shape(image_embedding)[1]],
          dtype=image_embedding.dtype)
      image_embedding = tf.transpose(image_embedding, [1, 0, 2])
      # encoder
      image_encoded = py_utils.NestedMap(
          input_embs=image_embedding, paddings=image_paddings)
      encoder_outputs = self.encoder.FPropTransformerLayers(
          theta.encoder, image_encoded)

    else:
      premise_ids = input_batch.premise_ids
      premise_paddings = input_batch.premise_paddings
      hypothesis_ids = input_batch.hypothesis_ids
      hypothesis_paddings = input_batch.hypothesis_paddings

      # encoder
      text_inputs = py_utils.NestedMap(
          ids=premise_ids, paddings=premise_paddings)
      text_encoded = self.encoder.FPropEmbeddings(theta.encoder, text_inputs)
      encoder_outputs = self.encoder.FPropTransformerLayers(
          theta.encoder, text_encoded)

    # decoder
    targets = py_utils.NestedMap(
        ids=hypothesis_ids, paddings=hypothesis_paddings)
    decoder_outputs = self.decoder.ComputePredictions(theta.decoder,
                                                      encoder_outputs, targets)

    classifier_input = self._extract_classifier_input(hypothesis_paddings,
                                                      decoder_outputs)
    return self._apply_classifier(theta, classifier_input)


class NLVR2Classification(GLUEClassification):
  """NLVR2 task of classification.

  Given two images and the sentence, we compute classifier input for each image
  paired with the sentence, and classify based on the concatenation of the two
  classifier inputs.
  """

  def _ComputeClassifierInput(self, theta, ids, paddings, image):
    image_embedding = self.image_embedding.FProp(theta.image_embedding, image)
    image_paddings = tf.zeros(
        [tf.shape(image_embedding)[0],
         tf.shape(image_embedding)[1]],
        dtype=image_embedding.dtype)
    image_embedding = tf.transpose(image_embedding, [1, 0, 2])

    image_encoded = py_utils.NestedMap(
        input_embs=image_embedding, paddings=image_paddings)
    encoder_outputs = self.encoder.FPropTransformerLayers(
        theta.encoder, image_encoded)

    targets = py_utils.NestedMap(ids=ids, paddings=paddings)
    decoder_outputs = self.decoder.ComputePredictions(theta.decoder,
                                                      encoder_outputs, targets)

    return self._extract_classifier_input(paddings, decoder_outputs)

  def ComputePredictions(self, theta, input_batch):
    ids = input_batch.ids
    paddings = input_batch.paddings
    image0 = input_batch.image0
    image1 = input_batch.image1

    img0_classifier_input = self._ComputeClassifierInput(
        theta, ids, paddings, image0)
    img1_classifier_input = self._ComputeClassifierInput(
        theta, ids, paddings, image1)

    classifier_input = tf.concat([img0_classifier_input, img1_classifier_input],
                                 axis=1)

    return self._apply_classifier(theta, classifier_input)


class ImageGenerationBaseTask(MultimodalBaseTask):
  """A base task for image generation.

  ImageGenerationBaseTask enables image generation metrics like Inception score
  and FID score by feeding generated images and original images into
  Inception V3 model and generating activations and logits for metrics
  calculation.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'enable_is_fid_metrics', True,
        'Whether or not to enable calculation of FID and Inception Score.')
    p.Define(
        'num_decoded_images_to_keep', None,
        'Number of decoded images per replica to keep in DecodeWithTheta. It '
        'is useful to limit the size of decode tensors to be within 2GB in '
        'decode_out_dict = sess.run(decode_tensors). If None, keep all '
        'images. Each subclass should implement the logic to slice all decoded '
        'images (e.g., originals, reconstructed, sampled images.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    if p.enable_is_fid_metrics:
      self.CreateChild('inception', tfhub_encoders.InceptionV3.Params())

  def ShrinkImagesDecodeOut(self, images_nmap):
    """Helper function to shrink decode out images.

    We perform converting to uint8 and keeping p.num_decoded_images_to_keep
    images if set.

    Args:
      images_nmap: a NestedMap only with keys of image tensors.

    Returns:
      a NestedMap with converted images of same keys as images_nmap.
    """
    out_nmap = py_utils.NestedMap()
    for k in images_nmap:
      images = images_nmap[k]
      images = py_utils.HasRank(images, 4)
      assert images.dtype == tf.float32
      out_nmap[k] = tf.image.convert_image_dtype(images, tf.uint8)

      num_samples = self.params.num_decoded_images_to_keep
      if num_samples is not None:
        out_nmap[k] = out_nmap[k][:num_samples]
    return out_nmap

  def CreateDecoderMetrics(self):
    decoder_metrics = {'num_samples_in_batch': metrics.AverageMetric()}
    if self.params.enable_is_fid_metrics:
      decoder_metrics.update({
          'inception_score': multimodal_metrics.InceptionScoreMetrics(),
          'fid_score': multimodal_metrics.FIDScoreMetrics(),
      })
    return decoder_metrics

  def MaybeAddInceptionDecodeOutput(self, theta, original_images,
                                    generated_images):
    """Optionally adds inception activations and logits to decoder outputs.

    Args:
      theta: A `py_utils.NestedMap` object of variables.
      original_images: [batch_size, h, w, 3] with value range in [0, 1].
      generated_images: [batch_size, h, w, 3] with value range in [0, 1].

    Returns:
      A NestedMap that is empty if p.enable_is_fid_metrics is False, otherwise,
        - inception_original_logits: [batch_size, 1008]
        - inception_original_activations: [batch_size, 2048]
        - inception_generated_logits: [batch_size, 1008]
        - inception_generated_activations: [batch_size, 2048]
    """
    if not self.params.enable_is_fid_metrics:
      return py_utils.NestedMap()

    original_out = self.inception.FProp(theta.inception, original_images)
    generated_out = self.inception.FProp(theta.inception, generated_images)
    return py_utils.NestedMap({
        'inception_original_logits': original_out.logits,
        'inception_original_activations': original_out.pool_3,
        'inception_generated_logits': generated_out.logits,
        'inception_generated_activations': generated_out.pool_3,
    })

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    if not self.params.enable_is_fid_metrics:
      return decode_out_dict

    decode_metrics_dict['inception_score'].Update(
        decode_out_dict['inception_generated_logits'])

    decode_metrics_dict['fid_score'].Update(
        decode_out_dict['inception_original_activations'],
        decode_out_dict['inception_generated_activations'])
    return decode_out_dict

  def DecodeFinalize(self, decode_finalize_args):
    # TODO(yonghui): write decoder output to cns.
    pass

  def Inference(self):
    subgraphs = dict()
    with tf.name_scope('inference'):
      with tf.name_scope('default'):
        subgraphs['default'] = self._InferenceSubgraph_Default()
      with tf.name_scope('default_tpu_8'):
        subgraphs['default_tpu'] = self._InferenceSubgraph_Default(
            use_tpu=True, batch_size=8)
    return subgraphs

  def _InferenceSubgraph_Default(self, use_tpu=False, batch_size=None):
    """Inference subgraph for SavedModel export."""
    records = tf.placeholder(tf.string, shape=[batch_size], name='tfrecords')
    batch = self.input_generator.ParseTFRecords(records)
    images = tf.identity(batch.image, name='images')
    batch.image = images

    rets = self.Decode(batch)
    if use_tpu:

      def _InferenceFn(*flat_batch):
        tpu_batch = tf.nest.pack_sequence_as(batch, flat_batch)
        return tf.nest.flatten(self.Decode(tpu_batch))

      flat_rets = py_utils_tpu.RewriteForMultiCore(
          _InferenceFn, core_ordinal_feed=-1, inputs=tf.nest.flatten(batch))
      rets = tf.nest.pack_sequence_as(rets, flat_rets)

    feeds = py_utils.NestedMap({'tfrecords': records, 'images': images})
    fetches = py_utils.NestedMap({
        k: rets[k] for k in ('original_images', 'reconstructed_images', 'codes')
    })
    return fetches, feeds


class DalleDVae(ImageGenerationBaseTask):
  """DALL-E DVae task."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('dvae_mdl', None,
             'The dVAE model for dalle image encoding and decoding.')
    # TODO(yonghui): Hyper-params here are directly taken from the paper. They
    # might need to be adjusted based on different training setups.
    p.Define(
        'temperature',
        schedule.CosineSchedule.Params().Set(
            initial_value=1.0, final_value=1.0 / 16.0, total_steps=150000),
        'Schedule for gumbel softmax temperature.')
    p.Define(
        'beta',
        schedule.CosineSchedule.Params().Set(
            initial_value=0.0, final_value=6.6, total_steps=5000),
        'Schedule for KL loss weight.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.dvae_mdl
    self.CreateChild('dvae_mdl', p.dvae_mdl)
    self.CreateChild('temperature', p.temperature)
    self.CreateChild('beta', p.beta)

  def ComputePredictions(self, theta, input_batch):
    """Compute predictions from multimodal inputs.

    Args:
      theta: A NestedMap of theta.
      input_batch: A NestedMap of input.

    Returns:
     A NestedMap containing the following elements.
       - scaled_kl: scaled kl loss.
       - reconstruction_loss: reconstruction loss.
       - decoded_images: the decoded img.
       - loss: the combined loss.
    """
    temperature = self.temperature.Value()
    beta = self.beta.Value()
    lingvo_summary_utils.scalar('temperature', temperature)
    lingvo_summary_utils.scalar('beta', beta)
    return self.dvae_mdl.FProp(theta.dvae_mdl, input_batch.image, temperature,
                               beta)

  def ComputeLoss(self, theta, predictions, input_batch):
    loss_weight = tf.constant(1.0, dtype=tf.float32)
    rets = {
        'loss': (predictions.loss, loss_weight),
        'scaled_kl': (predictions.scaled_kl, loss_weight),
        'codebook_pplx': (predictions.codebook_pplx, loss_weight),
        'codebook_entropy': (predictions.codebook_entropy, loss_weight),
        'reconstruction_loss': (predictions.reconstruction_loss, loss_weight)
    }
    return rets, {}

  def DecodeWithTheta(self, theta, input_batch):
    """Auto-encodes input_batch.image."""
    orig_img = input_batch.image

    img_code = self.dvae_mdl.EncodeImg(theta.dvae_mdl, orig_img)
    decoded = self.dvae_mdl.DecodeImg(theta.dvae_mdl, img_code)

    out = py_utils.NestedMap()
    out['codes'] = img_code
    out.update(self.MaybeAddInceptionDecodeOutput(theta, orig_img, decoded))
    out.update(
        self.ShrinkImagesDecodeOut(
            py_utils.NestedMap(
                original_images=orig_img, reconstructed_images=decoded)))
    return out

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    decode_out_dict = super().PostProcessDecodeOut(decode_out_dict,
                                                   decode_metrics_dict)
    original_images = decode_out_dict['original_images']
    reconstructed_images = decode_out_dict['reconstructed_images']
    assert original_images.dtype == reconstructed_images.dtype == np.uint8
    num_samples = decode_out_dict['codes'].shape[0]
    decode_metrics_dict['num_samples_in_batch'].Update(num_samples)

    padded_imgs = summary_utils.pad_concat_images(
        [original_images, reconstructed_images])
    decode_out_dict['img_summary'] = summary_utils.image_to_summary(
        padded_imgs, name='original_reconstructed')
    return decode_out_dict


class DalleDVae3D(DalleDVae):
  """DALL-E DVae 3D task."""

  def DecodeWithTheta(self, theta, input_batch):
    """Auto-encodes input_batch.image."""
    out = super().DecodeWithTheta(theta, input_batch)

    def _Reshape(video_tensor):
      shape = video_tensor.shape.as_list()
      new_shape = [shape[0] * shape[1]] + shape[2:]
      return tf.reshape(video_tensor, shape=new_shape)

    # Reshape video tensors to image tensors.
    out['original_images'] = _Reshape(out['original_images'])
    out['reconstructed_images'] = _Reshape(out['reconstructed_images'])
    return out


class VqGan(ImageGenerationBaseTask):
  """The VQ-GAN model.

  Following PyTorch impl: http://shortn/_QOHZwnMhNE.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('data_dim', None, 'The data dim (in_channels).')
    p.Define('latent_dim', None, 'The latent dim (z_channels)')
    p.Define('vector_quantizer', None, 'The vector quantizer params.')
    p.Define('discriminator', None, 'The discriminator params.')
    p.Define('loss', vqgan.Loss.Params(), 'The loss layer params.')
    p.Define(
        'laplace_eps', 0.1,
        'laplace_eps as used in log-laplace distribution if '
        'p.loss.log_laplace_loss_weight > 0.')
    # The default PyTorch initialization
    #   http://shortn/_YLtKGA78eW
    # kaiming_uniform(a=sqrt(5))
    #   http://shortn/_8p22sR3gWs
    # gain = calculate_gain('leaky_relu', sqrt(5)) = sqrt(1/3)
    #   http://shortn/_7oFXbLPtQC
    p.params_init = py_utils.WeightInit.UniformUnitScaling(
        scale=math.sqrt(1. / 3.))
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.loss is not None:
      self.CreateChild('loss', p.loss)
    self.CreateChild(
        'encoder',
        p.encoder.Copy().Set(data_dim=p.data_dim, latent_dim=p.latent_dim))
    # TODO(rpang): experiment with removing quant_conv and post_quant_conv,
    # since they are consecutive affine transformations with the last conv of
    # encoder and the first conv of decoder.
    vq_dim = p.vector_quantizer.latent_dim
    self.CreateChild(
        'quant_conv',
        layers.Conv2D.Params().Set(
            has_bias=True,
            filter_shape=(1, 1, p.latent_dim, vq_dim),
            filter_stride=(1, 1),
            padding='SAME'))
    self.CreateChild(
        'post_quant_conv',
        layers.Conv2D.Params().Set(
            has_bias=True,
            filter_shape=(1, 1, vq_dim, p.latent_dim),
            filter_stride=(1, 1),
            padding='SAME'))
    decoder_data_dim = p.data_dim
    if self.use_log_laplace:
      assert p.loss.l1_loss_weight == 0 and p.loss.l2_loss_weight == 0
      decoder_data_dim *= 2
    self.CreateChild(
        'decoder',
        p.decoder.Copy().Set(
            data_dim=decoder_data_dim, latent_dim=p.latent_dim))
    self.CreateChild('vector_quantizer', p.vector_quantizer)
    if p.discriminator is not None:
      self.CreateChild('discriminator',
                       p.discriminator.Copy().Set(data_dim=p.data_dim))

  @property
  def use_log_laplace(self):
    p = self.params
    return p.loss is not None and p.loss.log_laplace_loss_weight > 0

  def ComputePredictions(self, theta, input_batch):
    x = input_batch.image
    output_nmap = py_utils.NestedMap()

    with tf.name_scope('encoder'):
      latent = self.encoder.FProp(theta.encoder, x)
      z = self.quant_conv.FProp(theta.quant_conv, latent)
    quantized = self.vector_quantizer.FProp(theta.vector_quantizer, z)
    with tf.name_scope('decoder'):
      latent_q = self.post_quant_conv.FProp(theta.post_quant_conv,
                                            quantized.z_q)
      decoder_out = self.decoder.FProp(theta.decoder, latent_q)
      if self.use_log_laplace:
        output_nmap.decoded_mu = decoder_out[..., :3]
        output_nmap.decoded_sigma = decoder_out[..., 3:]
        output_nmap.log_laplace_inputs = objectives.log_laplace_preprocess(
            input_batch.image * 0.5 + 0.5, self.params.laplace_eps)
        reconstructed = tf.math.sigmoid(output_nmap.decoded_mu)
        reconstructed = objectives.log_laplace_postprocess(
            reconstructed, self.params.laplace_eps) * 2.0 - 1.0
      else:
        reconstructed = decoder_out

    output_nmap.reconstructed = reconstructed
    output_nmap.loss_q = quantized.loss
    output_nmap.codebook_pplx = quantized.codebook_pplx
    output_nmap.codebook_entropy = quantized.codebook_entropy

    if 'discriminator' in self.children:
      with tf.name_scope('discriminator'):
        logits_real = self.discriminator.FProp(theta.discriminator, x)
        logits_fake = self.discriminator.FProp(theta.discriminator,
                                               reconstructed)
      output_nmap.r1_gradient_penalty = objectives.gan_gradients_penalty_loss(
          inputs=x, logits=logits_real)
      output_nmap.logits_real = logits_real
      output_nmap.logits_fake = logits_fake
    return output_nmap

  def ComputeLoss(self, theta, predictions, input_batch):
    loss_weight = tf.constant(1.0, dtype=tf.float32)
    losses = self.children['loss'].FProp(
        theta.loss,
        predictions=predictions,
        input_batch=input_batch,
        decoder_last_layer_weights=self.decoder.LastLayerWeights(theta.decoder))
    loss_metrics = {f'loss/{k}': (v, loss_weight) for k, v in losses.items()}
    # 'loss' is required by base_model. The actual learners will use both
    # 'loss/generative' and 'loss/discriminative'.
    loss_metrics['loss'] = loss_metrics['loss/generative']
    loss_metrics['codebook_pplx'] = (predictions.codebook_pplx, loss_weight)
    loss_metrics['codebook_entropy'] = (predictions.codebook_entropy,
                                        loss_weight)
    return loss_metrics, {}

  def EncodeImg(self, theta, img_inputs):
    """Encodes given images.

    Args:
      theta: weight params for this layer and its sub-layers.
      img_inputs: the input image, of shape [b, h, w, 3], normalized in range
        [0, 1].

    Returns:
      The encoding token ids of input images, of shape
        [b, h / reduction_factor, w / reduction_factor].
    """
    # Converts img_inputs from range [0, 1] to [-1, 1].
    img_inputs = img_inputs * 2 - 1.0
    latent = self.encoder.FProp(theta.encoder, img_inputs)
    z = self.quant_conv.FProp(theta.quant_conv, latent)
    quantized = self.vector_quantizer.FProp(theta.vector_quantizer, z)
    return quantized.codes

  def DecodeImg(self, theta, img_code):
    """Decodes img_code back into imgs.

    Args:
      theta: weight params for this layer and its sub-layers.
      img_code: the img code, of shape [b, h / reduction_factor, w /
        reduction_factor].

    Returns:
      Decoded img of shape [b, h, w, 3] of range [0, 1].
    """
    decoded = self.DecodeWithCodes(theta, img_code)
    return decoded['reconstructed_images']

  def DecodeWithCodes(self, theta, codes):
    """Generates images [B, H, W, 3] from codes, integers of [B, H, W]."""
    with tf.name_scope('decoder'):
      z_q = self.vector_quantizer.Lookup(theta.vector_quantizer, codes)
      latent_q = self.post_quant_conv.FProp(theta.post_quant_conv, z_q)
      decoder_out = self.decoder.FProp(theta.decoder, latent_q)
      if self.use_log_laplace:
        reconstructed = tf.math.sigmoid(decoder_out[..., :3])
        reconstructed = objectives.log_laplace_postprocess(
            reconstructed, self.params.laplace_eps) * 2.0 - 1.0
      else:
        reconstructed = decoder_out
    return {
        # Latent vectors: [B, latent_dim].
        'z_q':
            z_q,
        # [B, H, W, 3] in range [0, 1].
        'reconstructed_images':
            tf.clip_by_value((reconstructed + 1.0) * 0.5, 0., 1.),
    }

  def DecodeWithTheta(self, theta, input_batch):
    with tf.name_scope('dec'):
      x = input_batch.image
      with tf.name_scope('encoder'):
        latent = self.encoder.FProp(theta.encoder, x)
        z = self.quant_conv.FProp(theta.quant_conv, latent)
      quantized = self.vector_quantizer.FProp(theta.vector_quantizer, z)
      out = self.DecodeWithCodes(theta, quantized.codes)
      # Rescale values from [-1, 1] to [0, 1].
      out['codes'] = quantized.codes
      orig_img = (input_batch.image + 1.0) * 0.5
      out.update(
          self.MaybeAddInceptionDecodeOutput(theta, orig_img,
                                             out['reconstructed_images']))

      out.update(
          self.ShrinkImagesDecodeOut(
              py_utils.NestedMap(
                  original_images=orig_img,
                  reconstructed_images=out['reconstructed_images'],
                  z_q=quantized.z_q)))
      return out

  # TODO(rpang): refactor the summary logit to a common library.
  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    decode_out_dict = super().PostProcessDecodeOut(decode_out_dict,
                                                   decode_metrics_dict)
    original_images = decode_out_dict['original_images']
    reconstructed_images = decode_out_dict['reconstructed_images']
    assert original_images.dtype == reconstructed_images.dtype == np.uint8
    num_samples = decode_out_dict['codes'].shape[0]
    decode_metrics_dict['num_samples_in_batch'].Update(num_samples)

    padded_imgs = summary_utils.pad_concat_images(
        [original_images, reconstructed_images])
    decode_out_dict['img_summary'] = summary_utils.image_to_summary(
        padded_imgs, name='original_reconstructed')
    return decode_out_dict


class DalleLm(ImageGenerationBaseTask):
  """DALL-E Lm task."""

  @classmethod
  def Params(cls):
    p = super().Params()
    # TODO(jiahuiyu, weihan): Make rerank_mdl TfHub-absed or SavedModel-based.
    p.Define(
        'rerank_mdl', None,
        'None or an inherited class of ImageTextContrastiveTask as re-ranking '
        'model based on normalized alignment score (cosine distance).')
    p.Define('lm', dalle.DalleLm.Params(), 'Dalle LM params.')
    p.Define('num_hyps_per_beam', 8, 'Num of hyps per beam.')
    p.Define(
        'num_hyps_per_beam_to_keep', 4,
        'Num of top-k hyps per beam to keep. We sample num_hyps_per_beam '
        'first, re-rank all num_hyps_per_beam images, and keep top-k hyps to '
        'decoded outputs for post-processing.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    self.CreateChild('lm', p.lm)
    if p.rerank_mdl is not None:
      self.CreateChild('rerank_mdl', p.rerank_mdl)

  def ComputePredictions(self, theta, input_batch):
    """Compute predictions from multimodal inputs.

    Args:
      theta: A NestedMap of theta.
      input_batch: A NestedMap of input, containing the following elements: ids
        - [b, text_seq_len], the input text ids. labels - [b, text_seq_len], the
        text labels to predict. paddings - [b, text_seq_len], the paddings
        tensor for text sequence. image - [b, h, w, 3], the input images.

    Returns:
     A NestedMap containing the following elements.
       - loss: combined text seq loss and img seq loss.
       - text_lm_loss: loss on the text sequence.
       - img_lm_loss: loss on the img sequence.
    """

    # input_batch.ids are of shape [b, text_max_num, seq_len]
    def _AssertShape(x):
      _, text_max_num, _ = py_utils.GetShape(x)
      assert text_max_num == 1

    _AssertShape(input_batch.ids)
    _AssertShape(input_batch.labels)
    _AssertShape(input_batch.paddings)
    ids = tf.squeeze(input_batch.ids, 1)
    labels = tf.squeeze(input_batch.labels, 1)
    paddings = tf.squeeze(input_batch.paddings, 1)
    return self.lm.FProp(
        theta.lm,
        py_utils.NestedMap(
            image=input_batch.image,
            ids=ids,
            labels=labels,
            label_weights=1.0 - paddings,
            paddings=paddings,
            source_id=input_batch.Get('source_id', None)))

  def ComputeLoss(self, theta, predictions, input_batch):
    loss_weight = tf.constant(1.0, dtype=tf.float32)
    rets = {
        'loss': (predictions.loss, loss_weight),
        'text_lm_loss': (predictions.text_lm_loss, loss_weight),
        'img_lm_loss': (predictions.img_lm_loss, loss_weight),
        'img_tokens_entropy': (predictions.img_tokens_entropy, loss_weight),
        'img_tokens_pplx': (predictions.img_tokens_pplx, loss_weight)
    }
    return rets, {}

  def MaybeAddRerankMdlDecodeOutput(self, theta, input_batch, decoded_images):
    """Optionally adds alignment scores of text and images to decoder outputs.

    Args:
      theta: A `py_utils.NestedMap` object of variables.
      input_batch: A `py_utils.NestedMap` object of required fields:
        - ids: a tensor of shape [batch_size, 1, text_max_len].
        - labels: a tensor of shape [batch_size, 1, text_max_len].
        - paddings: a tensor of shape [batch_size, 1, text_max_len] as paddings.
      decoded_images: a tensor of shape [batch_size * num_hyps, h, w, c].

    Returns:
      A NestedMap that is empty if p.rerank_mdl is None, otherwise,
        - alignment_score: [batch_size, num_hyps]
    """
    p = self.params
    if p.rerank_mdl is None:
      return py_utils.NestedMap()

    # Deep copy and override input_batch.image to decoded_images.
    input_batch = tf.nest.map_structure(lambda x: x, input_batch)
    # TODO(jiahuiyu): Use [batch_size, num_hyps, h, w, c] instead.
    input_batch.image = decoded_images

    input_batch = self.rerank_mdl.ProcessInputBatch(theta.rerank_mdl,
                                                    input_batch)
    rerank_decoded = self.rerank_mdl.ComputePredictions(theta.rerank_mdl,
                                                        input_batch)
    txt_proj = rerank_decoded['text_projection']  # [B, 1, D]
    img_proj = rerank_decoded['image_projection']  # [B * H, D]

    bs = py_utils.GetShape(txt_proj)[0]
    img_proj = tf.reshape(img_proj, shape=(bs, p.num_hyps_per_beam, -1))
    return py_utils.NestedMap(
        alignment_score=tf.reduce_sum(img_proj * txt_proj, -1))

  def _ReorderTopKDecodedImages(self, decoded_images, alignment_score):
    """Reorders decoded images based on alignment scores and keep top-K outputs.

    Args:
      decoded_images: A tensor of decoder outputs that has shape [bs * num_hyps,
        h, w, c].
      alignment_score: A tensor of float scores that has shape [bs, num_hyps].

    Returns:
      A NestedMap of three tensors:
        - reordered_decoded_images: reordered decoded_images by alignment_score
          with num_hyps reduced to p.num_hyps_per_beam_to_keep.
        - reordered_alignment_score:  scores corresponding to reordered decoded
          images.
        - top_1_decoded_images: top-1 decoded images by alignment scores.
    """
    p = self.params
    argsort = tf.argsort(alignment_score, axis=-1, direction='DESCENDING')
    _, h, w, c = py_utils.GetShape(decoded_images)
    decoded_images = tf.reshape(decoded_images,
                                [-1, p.num_hyps_per_beam, h, w, c])
    top_k = p.num_hyps_per_beam_to_keep
    reordered_alignment_score = tf.gather(
        alignment_score, argsort, axis=1, batch_dims=1)[:, :top_k]
    reordered_decoded_images = tf.gather(
        decoded_images, argsort, axis=1, batch_dims=1)[:, :top_k]
    top_1_decoded_images = reordered_decoded_images[:, 0]
    reordered_decoded_images = tf.reshape(reordered_decoded_images,
                                          [-1, h, w, c])
    return py_utils.NestedMap(
        reordered_decoded_images=reordered_decoded_images,
        reordered_alignment_score=reordered_alignment_score,
        top_1_decoded_images=top_1_decoded_images)

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the inference graph for eval decoding with theta."""
    p = self.params
    orig_img = input_batch.image

    # input_batch.ids are of shape [b, text_max_num, seq_len]
    def _AssertShape(x):
      _, text_max_num, _ = py_utils.GetShape(x)
      assert text_max_num == 1

    _AssertShape(input_batch.ids)
    _AssertShape(input_batch.paddings)
    ids = tf.squeeze(input_batch.ids, 1)
    paddings = tf.squeeze(input_batch.paddings, 1)
    # TODO(jiahuiyu): Return lm_decoder_out['decoded_images'] of shape
    # [bs, num_hyps, h, w, c] instead of [bs * num_hyps, h, w, c].
    lm_decoder_out = self.lm.Decode(
        theta.lm,
        py_utils.NestedMap(
            ids=ids,
            paddings=paddings,
            source_id=input_batch.Get('source_id', None)),
        num_hyps_per_beam=p.num_hyps_per_beam)
    recons_img = self.lm.AutoEncodeImg(theta.lm, orig_img)

    # Add the original image.
    out = py_utils.NestedMap()
    out['original_images'] = orig_img
    out['reconstructed_images'] = recons_img
    out['ids'] = ids
    out['paddings'] = paddings
    out.update(
        self.MaybeAddRerankMdlDecodeOutput(theta, input_batch,
                                           lm_decoder_out['decoded_images']))
    if p.rerank_mdl is not None:
      rerank_out = self._ReorderTopKDecodedImages(lm_decoder_out.decoded_images,
                                                  out.alignment_score)
      out['decoded_images'] = rerank_out.reordered_decoded_images
      out['alignment_score'] = rerank_out.reordered_alignment_score
      # Evaluates InceptionDecodeOutput with top_1_decoded_images.
      out.update(
          self.MaybeAddInceptionDecodeOutput(theta, orig_img,
                                             rerank_out.top_1_decoded_images))
    else:
      out['decoded_images'] = lm_decoder_out['decoded_images']
      out.update(
          self.MaybeAddInceptionDecodeOutput(
              theta, orig_img,
              lm_decoder_out['decoded_images'][::p.num_hyps_per_beam]))

    out.update(
        self.ShrinkImagesDecodeOut(
            py_utils.NestedMap(
                original_images=orig_img,
                reconstructed_images=out['reconstructed_images'],
                decoded_images=out['decoded_images'])))
    return out

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    decode_out_dict = super().PostProcessDecodeOut(decode_out_dict,
                                                   decode_metrics_dict)

    p = self.params
    orig_img = decode_out_dict['original_images']
    decoded_img = decode_out_dict['decoded_images']
    recons_img = decode_out_dict['reconstructed_images']
    assert orig_img.dtype == decoded_img.dtype == recons_img.dtype == np.uint8
    num_samples = decode_out_dict['ids'].shape[0]
    num_hyps_per_beam_to_keep = p.num_hyps_per_beam_to_keep
    assert orig_img.shape[0] * num_hyps_per_beam_to_keep == decoded_img.shape[0]
    decode_metrics_dict['num_samples_in_batch'].Update(num_samples)

    # Construct images to be visualized.
    bs, h, w, c = orig_img.shape

    decoded_img_tensor = decoded_img.reshape(bs, num_hyps_per_beam_to_keep, h,
                                             w, c)
    summary_imgs = [orig_img, recons_img]
    summary_imgs.extend([
        decoded_img_tensor[:, i, :, :, :]
        for i in range(num_hyps_per_beam_to_keep)
    ])
    padded_imgs = summary_utils.pad_concat_images(summary_imgs)

    # Calculate alignment score between batch text and decoded images.
    if p.rerank_mdl is not None:
      # Flatten to python floats as alignment scores to visualize.
      hyps_batch_score = np.transpose(decode_out_dict['alignment_score'])
      alignment_scores = [[f'{score:.3f}'
                           for score in batch_score]
                          for batch_score in hyps_batch_score.tolist()]
      score_images = [
          summary_utils.texts_to_images(batch_scores, 12, w + 4)
          for batch_scores in alignment_scores
      ]
    else:
      score_images = [np.zeros((bs, 12, w + 4, 3))] * num_hyps_per_beam_to_keep

    assert decode_out_dict['ids'].dtype == np.int32
    ids_py = decode_out_dict['ids']
    lens_py = np.sum(1.0 - decode_out_dict['paddings'], 1).astype(np.int32)
    txt_py = self.input.tokenizer.IdsToStringsPython(ids_py, lens_py)

    # Visualize texts as images with fixed size ((img_size + 4) * 2, 12).
    txt_image = summary_utils.texts_to_images(txt_py, 12,
                                              (orig_img.shape[1] + 4) * 2)

    # Concatenate all images of same batch id horizontally with texts and
    # scores layout as:
    # +---------------------------------------------------------------+
    # | <orig image> <dvae image> <decoded img 0> ... <decoded img N> |
    # | <input text, (12, 2*w)  > <score img 0  > ... <score img N  > |
    # +---------------------------------------------------------------+
    padded_txts = np.concatenate([txt_image] + score_images, axis=2)
    padded_imgs = np.concatenate([padded_imgs, padded_txts], axis=1)

    decode_out_dict['img_summary'] = summary_utils.image_to_summary(
        padded_imgs,
        name=('orig_reconstructed_decoded_0_to_'
              f'{p.num_hyps_per_beam_to_keep - 1}'))
    decode_out_dict['txt_summary'] = summary_utils.text_to_summary(
        txt_py, name='text')
    return decode_out_dict
