"""Encoder subclasses for babelfish.mt.encoder."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.tasks.mt import encoder


class TransformerEncoderSeparateFprop(encoder.TransformerEncoder):
  """A wrapper for Transformer that supports Fprop with Transformer layers only.

  This is to enable taking a sequence of embeddings (e.g. image embedding).
  """

  def FPropTransformerLayers(self, theta, input_batch):
    """Fprop transformer layers only.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      input_batch: A `.NestedMap` object containing: input_embs - The inputs
        tensor of shape [time, batch, dim]. paddings - The input paddings of
        shape [batch, time].

    Returns:
      A '.NestedMap' object containing:
        encoded - The encoded features of shape [time, batch, dim].
        padding - The encoded features' padding of shape [time, batch].
    """
    p = self.params

    # [time, batch, dim]
    transformer_input = input_batch.input_embs
    paddings = tf.cast(
        tf.transpose(input_batch.paddings), py_utils.FPropDtype(p))

    # TODO(ziruiw): Take care of Masked token for pretraining.
    encoded, padding, _ = self.transformer_stack.FProp(theta.transformer_stack,
                                                       transformer_input,
                                                       paddings)
    return py_utils.NestedMap(encoded=encoded, padding=padding)

  def FPropEmbeddings(self, theta, input_batch):
    """Fprop embedding layers only.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      input_batch: A `.NestedMap` object containing: ids - The inputs tensor of
        shape [batch, time]. paddings - The ids' paddings of shape [batch,
        time].

    Returns:
      A '.NestedMap' object containing:
        input_embs - The input embedding to the transformer layers of shape
        [time, batch, dim].
        padding - The input padding of shape [batch, time].
    """

    p = self.params
    if p.packed_input:
      raise ValueError('not supported for now')

    with tf.name_scope(p.name):
      input_ids = py_utils.with_dependencies([
          py_utils.assert_shape_match(
              tf.shape(input_batch.ids), tf.shape(input_batch.paddings)),
          py_utils.assert_equal(tf.rank(input_batch.ids), 2)
      ], input_batch.ids)

      if (not py_utils.use_tpu() and
          tf.flags.FLAGS.transformer_encoder_truncates_inputs):
        max_seq_length = tf.cast(
            tf.reduce_max(tf.reduce_sum(1.0 - input_batch.paddings, 1)),
            tf.int32)
        paddings = py_utils.with_dependencies([
            py_utils.assert_equal(
                tf.constant(True, tf.bool),
                tf.reduce_all(input_batch.paddings[:, max_seq_length:] > 0.5))
        ], input_batch.paddings)
        input_ids = input_ids[:, :max_seq_length]
        paddings = paddings[:, :max_seq_length]
      else:
        paddings = input_batch.paddings

      max_time = tf.shape(input_ids)[1]

      # Input token embeddings + positional embeddings
      if not p.shared_emb:
        input_embs = self.token_emb.EmbLookup(theta.token_emb,
                                              tf.reshape(input_ids, [-1]))
      else:
        input_embs = self.softmax.EmbLookup(theta.softmax,
                                            tf.reshape(input_ids, [-1]))

      input_embs = tf.reshape(input_embs,
                              [-1, max_time, p.token_emb.embedding_dim])

      position_embs = self.position_emb.FProp(theta.position_emb, max_time)
      position_embs = tf.reshape(position_embs,
                                 [1, max_time, p.token_emb.embedding_dim])
      input_embs += position_embs
      if p.task_emb:
        input_embs += self.task_emb.EmbLookup(theta.task_emb,
                                              input_batch.task_ids)

      if p.model_dim != p.token_emb.embedding_dim:
        input_embs = self.emb_proj.FProp(theta.emb_proj, input_embs)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

      # [time, batch, dim]
      transformer_input = tf.transpose(input_embs, [1, 0, 2])
    return py_utils.NestedMap(input_embs=transformer_input, paddings=paddings)


class TransformerBatchMajorEncoderSeparateFprop(
    encoder.TransformerBatchMajorEncoder):
  """A batch-major Transformer that supports Fprop with Transformer layers only.

  This is to enable taking a sequence of embeddings (e.g. image embedding).
  """

  def FPropTransformerLayers(self, theta, input_batch):
    """Fprop transformer layers only.

    Ref: google3/third_party/py/lingvo/tasks/mt/encoder.py

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      input_batch: A `.NestedMap` object containing: input_embs - The inputs
        tensor of shape [time, batch, dim]. paddings - The input paddings of
        shape [time, batch] or [batch, time].

    Returns:
      A '.NestedMap' object containing:
        encoded - The encoded features of shape [time, batch, dim] or [batch,
          time, dim], depending p.output_data_format.
        padding - The encoded features' padding of shape [time, batch] or
          [batch, time].
    """

    p = self.params
    # [time, batch, dim]
    transformer_input = input_batch.input_embs
    paddings = input_batch.paddings

    transformer_input = tf.transpose(transformer_input, [1, 0, 2])
    
    shape = py_utils.GetShape(transformer_input)
    batch_size = shape[0]
    seq_len = shape[1]
    paddings = tf.transpose(paddings)
    paddings = tf.reshape(paddings, [batch_size, seq_len])

    segment_mask = tf.zeros([batch_size, 1, seq_len, seq_len])
    encoded, padding = self.transformer_stack.FProp(theta.transformer_stack,
                                                    transformer_input, paddings,
                                                    segment_mask)

    if p.output_data_format == 'TBC':
      encoded = tf.transpose(encoded, [1, 0, 2])  # [time, batch, dim]
      padding = tf.transpose(padding)  # [time, batch]
    return py_utils.NestedMap(encoded=encoded, padding=padding)

  def FPropEmbeddings(self, theta, input_batch):
    """Fprop embedding layers only.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      input_batch: A `.NestedMap` object containing: ids - The inputs tensor of
        shape [batch, time]. paddings - The ids' paddings of shape [batch,
        time].

    Returns:
      A '.NestedMap' object containing:
        input_embs - The input embedding to the transformer layers of shape
        [time, batch, dim].
        padding - The input padding of shape [time, batch].
    """

    p = self.params
    if p.packed_input:
      raise ValueError('not supported for now')

    p = self.params
    with tf.name_scope(p.name):
      # [batch, time]
      input_ids = input_batch.ids
      # [batch, time]
      paddings = input_batch.paddings

      batch = py_utils.GetShape(input_ids)[0]
      time = py_utils.GetShape(input_ids)[1]

      # Embedding layer.
      # [batch, time, dim]
      if not p.shared_emb:
        input_embs = self.token_emb.EmbLookup(theta.token_emb, input_ids)
      else:
        input_embs = self.softmax.EmbLookup(theta.softmax, input_ids)

      # [1, time, dim]
      position_embs = tf.expand_dims(
          self.position_emb.FProp(theta.position_emb, time), 0)

      # [batch, time, dim]
      input_embs += position_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)
        paddings = tf.cast(paddings, p.input_dropout_tpl.fprop_dtype)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
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

class TransformerBatchMajorEncoderMixedSeparateFprop(
    encoder.TransformerBatchMajorEncoderMixed):
  """A batch-major Transformer that supports Fprop with Transformer layers only.

  This is to enable taking a sequence of embeddings (e.g. image embedding).
  """

  def FPropTransformerLayers(self, theta, input_batch):
    """Fprop transformer layers only.

    Ref: google3/third_party/py/lingvo/tasks/mt/encoder.py

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      input_batch: A `.NestedMap` object containing: input_embs - The inputs
        tensor of shape [time, batch, dim]. paddings - The input paddings of
        shape [time, batch] or [batch, time].

    Returns:
      A '.NestedMap' object containing:
        encoded - The encoded features of shape [time, batch, dim] or [batch,
          time, dim], depending p.output_data_format.
        padding - The encoded features' padding of shape [time, batch] or
          [batch, time].
    """

    p = self.params
    # [time, batch, dim]
    transformer_input = input_batch.input_embs
    paddings = input_batch.paddings

    transformer_input = tf.transpose(transformer_input, [1, 0, 2])

    shape = py_utils.GetShape(transformer_input)
    batch_size = shape[0]
    seq_len = shape[1]
    paddings = tf.transpose(paddings)
    paddings = tf.reshape(paddings, [batch_size, seq_len])

    segment_mask = tf.zeros([batch_size, 1, seq_len, seq_len])
    encoded, padding = self.transformer_stack.FProp(theta.transformer_stack,
                                                    transformer_input, paddings,
                                                    segment_mask)

    if p.output_data_format == 'TBC':
      encoded = tf.transpose(encoded, [1, 0, 2])  # [time, batch, dim]
      padding = tf.transpose(padding)  # [time, batch]
    return py_utils.NestedMap(encoded=encoded, padding=padding)

  def FPropEmbeddings(self, theta, input_batch):
    """Fprop embedding layers only.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      input_batch: A `.NestedMap` object containing: ids - The inputs tensor of
        shape [batch, time]. paddings - The ids' paddings of shape [batch,
        time].

    Returns:
      A '.NestedMap' object containing:
        input_embs - The input embedding to the transformer layers of shape
        [time, batch, dim].
        padding - The input padding of shape [time, batch].
    """

    p = self.params
    if p.packed_input:
      raise ValueError('not supported for now')

    p = self.params
    with tf.name_scope(p.name):
      # [batch, time]
      input_ids = input_batch.ids
      # [batch, time]
      paddings = input_batch.paddings

      batch = py_utils.GetShape(input_ids)[0]
      time = py_utils.GetShape(input_ids)[1]

      # Embedding layer.
      # [batch, time, dim]
      if not p.shared_emb and not p.shared_emb_ex:
        input_embs = self.token_emb.EmbLookup(theta.token_emb, input_ids)
      else:
        input_embs = self.softmax.EmbLookup(theta.softmax, input_ids)

      # [1, time, dim]
      position_embs = tf.expand_dims(
          self.position_emb.FProp(theta.position_emb, time), 0)

      # [batch, time, dim]
      input_embs += position_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)
        paddings = tf.cast(paddings, p.input_dropout_tpl.fprop_dtype)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
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
