"""Decoder subclasses for babelfish.mt.decoder."""

from lingvo import compat as tf
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import scatter_update
from lingvo.tasks.mt import decoder

from google3.learning.brain.research.babelfish.multimodal import layers
from google3.learning.brain.research.babelfish.prefix_lm import model_utils


class PrefixTransformerDecoder(decoder.TransformerDecoder):
  """Transformer decoder that supports decoding with prefix."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('prefix_len', 0,
             'Max prefix length used in decoding. Disabled if set to 0.')

    p.trans_tpl.cls = layers.PrefixTransformerLayer
    p.trans_tpl.tr_atten_tpl.cls = layers.PrefixTransformerAttentionLayer
    return p

  def __init__(self, params):
    assert params.trans_tpl.cls == layers.PrefixTransformerLayer
    super().__init__(params)

  def _FProp(self, theta, encoder_outputs, targets):
    # TODO(ziruiw): Implement FProp with prefix in decoder.
    if self.params.prefix_len > 0:
      tf.logging.warning('FProp with prefix in decoder is not implemented!'
                         'Calling FProp will fall back to generic decoder!')

    return super()._FProp(theta, encoder_outputs, targets)

  def AddExtraDecodingInfo(self, encoder_outputs, targets):
    """Adds extra decoding information to encoded_outputs.

    For PrefixDecoder, targets must contain prefix_ids and prefix_paddings
    with shape [bz, prefix_len].

    Args:
      encoder_outputs: a NestedMap computed by encoder.
      targets: a NestedMap containing target input fields.

    Returns:
      encoder_ouputs with prefix ids and paddings to be delivered to decoder.
    """
    p = self.params
    if p.prefix_len <= 0:
      return super().AddExtraDecodingInfo(encoder_outputs, targets)

    if not ('prefix_ids' in targets and 'prefix_paddings' in targets):
      raise ValueError('targets must contain prefix_ids and prefix_paddings!')

    # Set the last prefix_ids as init_step_ids, use sos_id if empty.
    prefix_len = tf.cast(
        tf.reduce_sum(1 - targets.prefix_paddings, axis=1),
        targets.prefix_ids.dtype)
    prefix_one_hot = tf.one_hot(
        prefix_len - 1,
        py_utils.GetShape(targets.prefix_ids)[1],
        dtype=targets.prefix_ids.dtype)

    encoder_outputs['init_step_ids'] = tf.reduce_sum(
        prefix_one_hot * targets.prefix_ids, axis=1)

    nonempty_prefix = tf.cast(tf.greater(prefix_len, 0), prefix_one_hot.dtype)
    sos_prefix = tf.ones_like(encoder_outputs.init_step_ids) * p.target_sos_id
    encoder_outputs[
        'init_step_ids'] = nonempty_prefix * encoder_outputs.init_step_ids + (
            1 - nonempty_prefix) * sos_prefix

    # [batch, prefix_len]
    encoder_outputs['prefix_ids'] = py_utils.HasShape(
        targets.prefix_ids,
        [py_utils.GetShape(targets.prefix_ids)[0], p.prefix_len])
    encoder_outputs['prefix_paddings'] = py_utils.HasShape(
        targets.prefix_paddings,
        [py_utils.GetShape(targets.prefix_ids)[0], p.prefix_len])

    return encoder_outputs

  def _InitBeamSearchStateCallback(self, theta, encoder_outputs,
                                   num_hyps_per_beam):
    """Returns initial beams search states.

    For PrefixDecoder, we FProp prefix sequences to set as the prefix_states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder.
      num_hyps_per_beam: An int, number hyps to keep for source sentence.

    Returns:
      A tuple (initial_results, states).
        initial_results: a `.NestedMap` of initial results.
          atten_probs:
            The initial attention probs, of shape [tgt_batch, src_len].
        states: a `.NestedMap` of initial model states.
          source_encs:
            A tensor of shape [src_batch, src_len, source_dim].
          source_paddings:
            A tensor of shape [src_batch, src_len].
          time_step:
            A scalar, the initial decode step set as prefix_len.

    """
    p = self.params
    initial_results, states = super()._InitBeamSearchStateCallback(
        theta, encoder_outputs, num_hyps_per_beam)
    if p.prefix_len <= 0:
      return initial_results, states

    if p.beam_search.name == 'tpu_beam_search':
      seq_len = p.target_seq_len + p.prefix_len
    else:
      seq_len = 0

    prefix_states = py_utils.NestedMap()
    for layer in range(p.num_trans_layers):
      prefix_states['layer_%d' % layer] = py_utils.NestedMap({
          'key':
              tf.zeros([
                  seq_len,
                  py_utils.GetShape(
                      states.prefix_states['layer_%d' % layer].key)[1],
                  py_utils.GetShape(
                      states.prefix_states['layer_%d' % layer].key)[2]
              ],
                       dtype=states.prefix_states['layer_%d' %
                                                  layer].key.dtype),
          'value':
              tf.zeros([
                  seq_len,
                  py_utils.GetShape(
                      states.prefix_states['layer_%d' % layer].value)[1],
                  py_utils.GetShape(
                      states.prefix_states['layer_%d' % layer].value)[2]
              ],
                       dtype=states.prefix_states['layer_%d' %
                                                  layer].value.dtype),
      })

    # Always set step_ids for prefix mode.
    initial_results['step_ids'] = tf.expand_dims(
        self._ExpandToNumHyps(encoder_outputs.init_step_ids, num_hyps_per_beam),
        1)

    prefix_ids = encoder_outputs['prefix_ids']
    prefix_paddings = encoder_outputs['prefix_paddings']

    prefix_ids = tf.tile(input=prefix_ids, multiples=[num_hyps_per_beam, 1])
    prefix_paddings = tf.tile(
        input=prefix_paddings, multiples=[num_hyps_per_beam, 1])

    # Fprop prefix through decoder to init prefix_states.
    # Ref: google3/third_party/py/lingvo/tasks/mt/decoder.py
    source_encs = encoder_outputs.encoded
    source_paddings = encoder_outputs.padding
    time, batch = py_utils.GetShape(source_paddings, 2)
    if p.is_transparent:
      if self.do_eval:
        source_encs = py_utils.HasShape(
            source_encs, [time, batch, p.source_dim, p.num_trans_layers])
        source_encs = tf.unstack(source_encs, axis=3)
      else:
        assert isinstance(source_encs, list)
        assert len(source_encs) == p.num_trans_layers
        for i in range(p.num_trans_layers):
          source_encs[i] = py_utils.HasShape(source_encs[i],
                                             [time, batch, p.source_dim])
    else:
      source_encs = py_utils.HasShape(source_encs, [time, batch, p.source_dim])
      source_encs = [source_encs] * p.num_trans_layers
    with tf.name_scope(p.name):
      # [batch, time]
      target_ids = prefix_ids
      # [time, batch]
      target_paddings = tf.transpose(prefix_paddings)

      # Embedding layer
      # [batch, time, model_dim]
      if not self._share_sm_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, target_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, target_ids)

      target_time = py_utils.GetShape(target_ids)[1]

      # [1, time, model_dim]
      posit_embs = tf.expand_dims(
          self.position_emb.FProp(theta.position_emb, target_time), 0)

      # [time, batch, model_dim]
      input_embs = token_embs + posit_embs
      input_embs = tf.transpose(input_embs, [1, 0, 2])
      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

      layer_in = input_embs
      for i, (layer, layer_theta) in enumerate(zip(self.trans, theta.trans)):
        extra_kwargs = dict()
        if isinstance(layer, layers_with_attention.TransformerWithContextLayer):
          extra_kwargs['tertiary_vecs'] = encoder_outputs.context_encoded
          extra_kwargs['tertiary_paddings'] = encoder_outputs.context_padding

        layer_out, _ = layer.FProp(
            layer_theta,
            layer_in,
            target_paddings,
            tf.tile(input=source_encs[i], multiples=[num_hyps_per_beam, 1, 1]),
            tf.tile(input=source_paddings, multiples=[num_hyps_per_beam, 1]),
            source_segment_id=None,
            aux_segment_id=None,
            atten_idx=None,
            **extra_kwargs)

        # [time, batch, model_dim], apply layer_norm if needed.
        if layer.params.tr_atten_tpl.pre_layer_norm:
          layer_in = layer.self_atten.layer_norm.FProp(
              layer_theta.self_atten.layer_norm, layer_in)
        # Set prefix_states for each layer.
        for t in range(p.prefix_len):
          cached_packed_src = py_utils.NestedMap(
              source_vecs=prefix_states['layer_%d' % i].key,
              source_contexts=prefix_states['layer_%d' % i].value,
              source_padding=None,
              source_segment_id=None)
          extended_packed_src = layer.self_atten.atten.ExtendSourcePacked(
              layer_theta.self_atten.atten,
              layer_in[t, :, :],
              layer_in[t, :, :],
              None,
              None,
              cached_packed_src,
              t=t if p.beam_search.name == 'tpu_beam_search' else None)
          prefix_states['layer_%d' % i].key = extended_packed_src.source_vecs
          prefix_states['layer_%d' %
                        i].value = extended_packed_src.source_contexts

        layer_in = layer_out

    return initial_results, py_utils.NestedMap({
        'prefix_states': prefix_states,
        'time_step': tf.constant(p.prefix_len)
    })

  def ExtendStep(self, theta, encoder_outputs, new_ids, t, prefix_states):
    """Extend prefix as represented by `prefix_states` by one more step.

    For PrefixDecoder, t starts at position = prefix_len so we need to adjust
    it back to its actual position in the sequence.

    Ref: google3/third_party/py/lingvo/tasks/mt/decoder.py

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder, containing:
        - encoded: source encoding, of shape [time, batch, depth]. Can be [time,
          bs, depth, num_trans_layers] if is_transparent is set.
        - padding: source encoding's padding, of shape [time, batch].
      new_ids: new input ids, of shape [batch].
      t: a scalar, the current time step, 0-based.
      prefix_states: a `.NestedMap` representing the prefix that has already
        been decoded.

    Returns:
      A tuple (last_decoder_out, prefix_states, atten_probs), where
      last_decoder_out is the output of the last decoder layer of
      shape [batch, model_dim], `prefix_states` is the update prefix states,
      and atten_probs contains attention in shape [batch, src_len] for the
      given target position.
    """
    p = self.params
    if p.prefix_len <= 0:
      return super().ExtendStep(theta, encoder_outputs, new_ids, t,
                                prefix_states)

    source_paddings = encoder_outputs.padding
    time, batch = py_utils.GetShape(source_paddings, 2)
    if p.is_transparent:
      source_encs = py_utils.HasShape(
          encoder_outputs.encoded,
          [time, batch, p.source_dim, p.num_trans_layers])
      source_encs = tf.unstack(source_encs, axis=3)
    else:
      source_encs = py_utils.HasShape(encoder_outputs.encoded,
                                      [time, batch, p.source_dim])
      source_encs = [source_encs] * p.num_trans_layers
    with tf.name_scope(p.name):
      # Embedding layer
      # [batch, model_dim]
      if not self._share_sm_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, new_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, new_ids)

      if p.zero_token_embs_first_time_step:
        # For models that do not use an explicit start-of-sequence token
        # with associated embedding, but instead use zeros.
        zeros = tf.zeros_like(token_embs)
        token_embs = tf.cond(tf.equal(t, 0), lambda: zeros, lambda: token_embs)

      # Infer num_hyps_per_beam: new_ids has orig_batch_size * num_hyps_per_beam
      # source_paddings has orig_batch_size.
      num_hyps_per_beam = tf.div(
          py_utils.GetShape(new_ids)[0],
          py_utils.GetShape(source_paddings)[1])

      # Different to ordinary ExtendStep starts here: we adjust position info t
      # based on the used prefix token, e.g. if len(prefix) = 4, t = 4, then the
      # correct t to use  should be t - max_prefix_len + (len(prefix) - 1) = 3
      # for positional embedding, if max_prefix_len = 4.
      t = py_utils.with_dependencies([
          py_utils.assert_between(t, p.prefix_len,
                                  p.prefix_len + p.target_seq_len)
      ], t)
      prefix_len = tf.cast(
          tf.reduce_sum(1 - encoder_outputs['prefix_paddings'], axis=1),
          t.dtype)
      # If t >= prefix_len, adjust the t to the actual position.
      adjusted_t = tf.cond(
          t >= p.prefix_len, lambda: t - p.prefix_len + prefix_len - 1,
          lambda: tf.ones_like(prefix_len - 1, dtype=prefix_len.dtype) * t)
      adjusted_t = self._ExpandToNumHyps(adjusted_t, num_hyps_per_beam)
      adjusted_t_one_hot = tf.one_hot(adjusted_t,
                                      p.target_seq_len + p.prefix_len)
      posit_embs = tf.einsum(
          'ij,jk->ik', adjusted_t_one_hot,
          self.position_emb.FProp(theta.position_emb,
                                  p.target_seq_len + p.prefix_len))

      input_embs = token_embs + posit_embs

      atten_idx = None
      if p.task_emb:
        task_ids = self._ExpandToNumHyps(encoder_outputs.target_task_ids,
                                         num_hyps_per_beam)
        if p.use_lang_dependent_atten:
          atten_idx = task_ids
        input_embs += self.task_emb.EmbLookup(theta.task_emb, task_ids)

      if p.model_dim != self._token_emb_dim:
        input_embs = self.emb_proj.FProp(theta.emb_proj, input_embs)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      # Make a copy of the input.
      out_prefix_states = prefix_states.Pack(prefix_states.Flatten())

      layer_in = input_embs

      # Infer true source encoder length from the padding.
      src_enc_len = tf.reduce_sum(1 - source_paddings, axis=0)

      # Need to expand src_enc_len to reflect multiple hypotheses.
      src_enc_len = self._ExpandToNumHyps(src_enc_len, num_hyps_per_beam)

      atten_probs = []

      # Example: if prefix is [t0 t1 pad] and t = 3, padding should be [0 1 1 0]
      # since t1 is used as the first infeed step_id of beam search.
      prefix_len = self._ExpandToNumHyps(prefix_len, num_hyps_per_beam)
      last_prefix_mask = tf.one_hot(
          prefix_len - 1,
          py_utils.GetShape(encoder_outputs['prefix_paddings'])[1],
          dtype=encoder_outputs['prefix_paddings'].dtype)
      prefix_paddings = tf.tile(
          input=encoder_outputs['prefix_paddings'],
          multiples=[num_hyps_per_beam, 1]) + last_prefix_mask
      beam_paddings = tf.zeros(
          [py_utils.GetShape(prefix_paddings)[0], t - p.prefix_len + 1],
          dtype=prefix_paddings.dtype)
      per_step_source_padding = tf.concat([prefix_paddings, beam_paddings],
                                          axis=1)
      if p.beam_search.name == 'tpu_beam_search':
        per_step_source_padding = py_utils.PadOrTrimTo(
            per_step_source_padding, [
                py_utils.GetShape(per_step_source_padding)[0],
                p.prefix_len + p.target_seq_len
            ],
            pad_val=1)

      for i, (layer, layer_theta) in enumerate(zip(self.trans, theta.trans)):
        extra_kwargs = dict()
        if isinstance(layer, layers_with_attention.TransformerWithContextLayer):
          # If the encoder contains encodings for the context and the
          # transformer layer in the decoder is able to attend to it, we pass
          # them to the transformer layer.
          extra_kwargs['tertiary_vecs'] = encoder_outputs.context_encoded
          extra_kwargs['tertiary_paddings'] = encoder_outputs.context_padding
        # [time, batch, model_dim]
        layer_prefix_states = prefix_states['layer_%i' % i]
        layer_out, probs, updated_prefix_states = layer.ExtendStep(
            layer_theta,
            layer_in,
            layer_prefix_states,
            source_encs[i],
            source_paddings,
            t=t if p.beam_search.name == 'tpu_beam_search' else None,
            atten_idx=atten_idx,
            per_step_source_padding=per_step_source_padding,
            **extra_kwargs)
        out_prefix_states['layer_%i' % i] = updated_prefix_states
        layer_in = layer_out
        # Enforce shape: [batch, src_len]
        probs = tf.squeeze(probs, [0])
        # Remove attention weight on last (EOS) token and re-normalize
        # so that last dimension sums to 1. See b/129097156.
        probs_3d = tf.expand_dims(probs, axis=1)
        probs_3d = self._RemoveEOSProbs(p, probs_3d, src_enc_len)
        probs = tf.squeeze(probs_3d, axis=1)

        atten_probs.append(probs)

      if p.ln_output:
        layer_out = self.layer_norm_out.FProp(theta.layer_norm_out, layer_out)

      # Aggregate per-layer attention probs.
      aggregated_atten_probs = tf.math.add_n(atten_probs) / len(atten_probs)
      return layer_out, out_prefix_states, aggregated_atten_probs


class PrefixTransformerBatchMajorDecoder(decoder.TransformerBatchMajorDecoder):
  """Transformer batch-major decoder that supports prefix inputs.

  It allows four settings by setting prefix_len and has_aux_atten params:
    (1) With Encoder & No Prefix: prefix_len = 0, has_aux_atten = True. This
    corresponds to generic encoder-decoder architectures.
    (2) With Encoder & With Prefix: prefix_len > 0, has_aux_atten = True. This
    corresponds to encoder-decoder architectures with additional prefix in the
    decoder.
    (3) No Encoder & With Prefix: prefix_len > 0, has_aux_atten = False. This
    corresponds to generic prefix LM with decoder-only architecture.
    (4) No Encoder & No Prefix: prefix_len = 0, has_aux_atten = False. This
    corresponds to generic LM with decoder-only architecture.

  In addition, it also supports customized transformer inputs such as images.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('prefix_len', 0,
             ('Max prefix length used in decoding. Disabled if set to 0.'
              'Notice that this is an indicator of prefix mode but this max len'
              'is used for decoding only, the user can set arbitrary prefix'
              'length during training.'))
    p.Define('has_aux_atten', True,
             'Set to enable attention over encoder outputs.')
    return p

  def __init__(self, params):
    if isinstance(params.trans_decoder_tpl, list):
      has_aux_atten = params.trans_decoder_tpl[0].has_aux_atten
    else:
      has_aux_atten = params.trans_decoder_tpl.has_aux_atten
    assert params.has_aux_atten == has_aux_atten
    super().__init__(params)

  def _FProp(self, theta, encoder_outputs, targets):
    p = self.params
    if not p.has_aux_atten:
      # Decoder-only architecture with no encoder atten.
      encoder_outputs = None
    if p.prefix_len > 0 and 'token_visibility' not in targets:
      raise ValueError('Must set token_visibility for prefix in decoder!')
    target_embeddings = self.FPropEmbeddings(theta, targets)
    target_embeddings['token_visibility'] = getattr(targets, 'token_visibility',
                                                    None)
    return self.FPropTransformerLayers(theta, encoder_outputs,
                                       target_embeddings)

  def FPropEmbeddings(self, theta, input_batch):
    """Compute the embeddings for text tokens.

    To enable prefix in the decoder, simply add prefix ids and paddings to
    input_batch and set token_visibility (default to autoregressive).
    For example:
      input_batch.ids = [1, 2, 0, 3, 4, 0]
      input_batch.paddings = [0, 0, 1, 0, 0, 1]
      input_batch.token_visibility = [0, 0, 0, 1, 2, 3]
      Then the first 3 tokens will then be treated as prefix tokens, where the
      position will become: [0, 1, 1, 2, 3, 3].

    Args:
      theta: A NestedMap containing weights' values of this layer and its
        children layers.
      input_batch: a NestedMap containing target input fields.

    Returns:
      A NestedMap containing Transformer input embeddings and paddings.
    """
    p = self.params
    with tf.name_scope(p.name):
      # [batch, target_time]
      target_ids = input_batch.ids
      target_paddings = input_batch.paddings

      # Embedding layer
      # [batch, target_time, dim]
      if not p.shared_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, target_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, target_ids)
      # [batch, target_time, dim]
      # Take care of prefix padding and adjust non-prefix position.
      positions = tf.math.cumsum(
          tf.cast(1. - target_paddings, tf.int32), axis=1) - 1
      posit_embs = self.position_emb.FPropWithPosition(theta.position_emb,
                                                       positions)

      # [batch, target_time, dim]
      input_embs = token_embs + posit_embs
      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)
        target_paddings = tf.cast(target_paddings,
                                  p.input_dropout_tpl.fprop_dtype)

      transformer_input = self.input_dropout.FProp(theta.input_dropout,
                                                   input_embs)
    return py_utils.NestedMap(
        transformer_input=transformer_input, paddings=target_paddings)

  def FPropTransformerLayers(self, theta, encoder_outputs, input_batch):
    """Compute the transformer outputs for input embeddings.

    To enable prefix in the decoder, simply set token_visibility.
    For example:
      input_batch.token_visibility = [0, 0, 1]
      => per_step_padding = [[0, 0, 1], [0, 0, 1], [0, 0, 0]]
      Then the first 2 tokens will then be treated as prefix and are allowed to
      attend to each other but not the last token, while the last one can attend
      to all tokens.

    Args:
      theta: A NestedMap containing weights' values of this layer and its
        children layers.
      encoder_outputs: a NestedMap computed by encoder.
      input_batch: a NestedMap containing target input fields.

    Returns:
      softmax_input: Tensor of shape [target_time, batch, dim].
    """
    p = self.params
    # [batch, target_time, dim]
    layer_in = input_batch.transformer_input
    target_paddings = input_batch.paddings

    per_step_padding = None
    token_visibility = getattr(input_batch, 'token_visibility', None)
    if token_visibility is not None:
      per_step_padding = model_utils.ComputePerStepPadding(
          token_visibility,
          tf.ones_like(token_visibility, dtype=token_visibility.dtype))

    # [batch, source_time, dim]
    aux_vec = None
    aux_paddings = None
    if encoder_outputs is not None:
      if not p.has_aux_atten:
        raise ValueError('Encoder output must be none for decoder only!')
      encoder_out_bm = self._MaybeTransposeEncoderOutputs(
          encoder_outputs, 'BTC')
      aux_vec = encoder_out_bm.encoded
      aux_paddings = encoder_out_bm.padding
      if aux_vec is None or aux_paddings is None:
        raise ValueError('Encoder output must be set for enc-dec architecture!')

    with tf.name_scope(p.name):
      for layer, layer_theta in zip(self.decoder_trans, theta.decoder_trans):
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
            per_step_padding_override=per_step_padding,
            segment_mask=None,
            aux_segment_mask=None)
        layer_in = layer_out

      if p.final_layer_norm:
        layer_out = self.final_ln.FProp(theta.final_ln, layer_out)
      if p.prediction_data_format == 'TBC':
        # Transpose the softmax_input to match the input requirement of
        # ComputePredictions.
        layer_out = tf.transpose(layer_out, [1, 0, 2])
    return layer_out

  def ComputeLoss(self, theta, predictions, targets):
    p = self.params
    if isinstance(predictions, py_utils.NestedMap):
      predictions = predictions.softmax_input
    target_time = py_utils.GetShape(predictions)[{
        'TBC': 0,
        'BTC': 1
    }.get(p.prediction_data_format)]
    batch = py_utils.GetShape(targets.labels)[0]

    # We allow passing labels for non-prefix only and pad here explicitly.
    # Notice that we pad BEFORE original inputs here, corresponding to prefix.
    targets.labels = py_utils.PadOrTrimTo(
        targets.labels, [batch, target_time], pad_after_contents=False)
    targets.weights = py_utils.PadOrTrimTo(
        targets.weights, [batch, target_time], pad_after_contents=False)
    targets.paddings = py_utils.PadOrTrimTo(
        targets.paddings, [batch, target_time], pad_after_contents=False)
    return super().ComputeLoss(theta, predictions, targets)

  def _TransposeBeamFormat(self,
                           tensor,
                           num_hyps_per_beam,
                           num_beam,
                           mode='HB2BH'):
    original_shape = py_utils.GetShape(tensor)

    if mode == 'HB2BH':
      input_shape = [num_hyps_per_beam, num_beam]
    elif mode == 'BH2HB':
      input_shape = [num_beam, num_hyps_per_beam]
    tensor = tf.reshape(tensor, input_shape + original_shape[1:])
    tensor = tf.transpose(tensor,
                          [1, 0] + list(range(2,
                                              len(original_shape) + 1)))
    tensor = tf.reshape(tensor, [-1] + original_shape[1:])
    return tensor

  def _MaybeTransposeField(self, input_map, output_map, field):
    if getattr(input_map, field, None) is None:
      output_map[field] = None
    else:
      output_map[field] = tf.transpose(input_map[field])

  def _MaybeTransposeEncoderOutputs(self, encoder_outputs, target_data_format):
    p = self.params
    if p.input_data_format == target_data_format:
      return encoder_outputs
    transposed = py_utils.NestedMap(
        encoded=tf.transpose(encoder_outputs.encoded, [1, 0, 2]),
        padding=tf.transpose(encoder_outputs.padding))
    self._MaybeTransposeField(encoder_outputs, transposed, 'prefix_ids')
    self._MaybeTransposeField(encoder_outputs, transposed, 'prefix_paddings')
    self._MaybeTransposeField(encoder_outputs, transposed, 'segment_id')
    return transposed

  def AddExtraDecodingInfo(self, encoder_outputs, targets):
    """Adds extra decoding information to encoded_outputs.

    For PrefixDecoder, targets must contain prefix_ids and prefix_paddings
    with shape [bz, prefix_len].

    Args:
      encoder_outputs: a NestedMap computed by encoder.
      targets: a NestedMap containing target input fields.

    Returns:
      encoder_ouputs with prefix ids and paddings to be delivered to decoder.
    """
    p = self.params
    if not p.has_aux_atten:
      if p.prefix_len <= 0:
        batch = py_utils.GetShape(targets.ids)[0]
      else:
        batch = py_utils.GetShape(targets.prefix_ids)[0]
      if p.input_data_format == 'TBC':
        encoded_shape = [1, batch, p.model_dim]
        padding_shape = [1, batch]
      else:
        encoded_shape = [batch, 1, p.model_dim]
        padding_shape = [batch, 1]
      # Dummy encoded and padding. We explicitly set to None for readability,
      # but it is ok to keep dummy values because they will be ignored.
      encoder_outputs = py_utils.NestedMap(
          encoded=tf.zeros(encoded_shape, dtype=tf.int32),
          padding=tf.zeros(padding_shape, dtype=tf.float32))

    if p.prefix_len <= 0:
      return super().AddExtraDecodingInfo(encoder_outputs, targets)

    if not ('prefix_ids' in targets and 'prefix_paddings' in targets):
      raise ValueError('targets must contain prefix_ids and prefix_paddings!')

    # Set the last prefix_ids as init_step_ids, use sos_id if empty.
    prefix_len = tf.cast(
        tf.reduce_sum(1 - targets.prefix_paddings, axis=1),
        targets.prefix_ids.dtype)
    prefix_one_hot = tf.one_hot(
        prefix_len - 1,
        py_utils.GetShape(targets.prefix_ids)[1],
        dtype=targets.prefix_ids.dtype)

    encoder_outputs['init_step_ids'] = tf.reduce_sum(
        prefix_one_hot * targets.prefix_ids, axis=1)

    nonempty_prefix = tf.cast(tf.greater(prefix_len, 0), prefix_one_hot.dtype)
    sos_prefix = tf.ones_like(encoder_outputs.init_step_ids) * p.target_sos_id
    encoder_outputs[
        'init_step_ids'] = nonempty_prefix * encoder_outputs.init_step_ids + (
            1 - nonempty_prefix) * sos_prefix

    # [batch, prefix_len]
    encoder_outputs['prefix_ids'] = py_utils.HasShape(
        targets.prefix_ids,
        [py_utils.GetShape(targets.prefix_ids)[0], p.prefix_len])
    encoder_outputs['prefix_paddings'] = py_utils.HasShape(
        targets.prefix_paddings,
        [py_utils.GetShape(targets.prefix_ids)[0], p.prefix_len])

    # Transpose to be consistent with input format.
    if p.input_data_format == 'TBC':
      encoder_outputs.prefix_ids = tf.transpose(encoder_outputs.prefix_ids)
      encoder_outputs.prefix_paddings = tf.transpose(
          encoder_outputs.prefix_paddings)
    return encoder_outputs

  def _UpdatePrefixStates(self, layer_id, layer, layer_theta, query_vec, t,
                          prefix_states):
    p = self.params

    target_batch = py_utils.GetShape(query_vec)[0]
    attention_dim = p.trans_decoder_tpl.tr_atten_tpl.input_dim
    num_heads = p.trans_decoder_tpl.tr_atten_tpl.num_heads
    per_head_dim = attention_dim // num_heads

    updated_prefix_states = prefix_states.DeepCopy()

    new_key_proj = layer.self_atten.atten.key.FProp(
        layer_theta.self_atten.atten.key, query_vec)
    new_value_proj = layer.self_atten.atten.value.FProp(
        layer_theta.self_atten.atten.value, query_vec)
    new_key_proj = tf.cast(
        tf.reshape(new_key_proj, [target_batch, num_heads, per_head_dim]),
        dtype=prefix_states['layer_%d' % layer_id].key.dtype)
    new_value_proj = tf.cast(
        tf.reshape(new_value_proj, [target_batch, num_heads, per_head_dim]),
        dtype=prefix_states['layer_%d' % layer_id].value.dtype)
    extended_key = scatter_update.Update(
        prefix_states['layer_%d' % layer_id].key, tf.convert_to_tensor(t),
        new_key_proj)
    extended_value = scatter_update.Update(
        prefix_states['layer_%d' % layer_id].value, tf.convert_to_tensor(t),
        new_value_proj)
    updated_prefix_states['layer_%d' % layer_id] = py_utils.NestedMap(
        key=extended_key, value=extended_value)
    return updated_prefix_states

  def _InitBeamSearchStateCallback(self, theta, encoder_outputs,
                                   num_hyps_per_beam):
    """Returns initial beams search states.

    For PrefixDecoder, we FProp prefix sequences to set as the prefix_states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: A '.NestedMap' object computed by encoder. * encoded -
        Source encoding of shape [source_time, source_batch, dim] or
        [source_batch, source_time, dim], depending on p.input_data_format. *
        paddings - Source encoding's padding of shape [source_time,
        source_batch] or [source_batch, source_time]. * prefix_ids - Prefix Ids
        of shape [source_batch, prefix_len]. * prefix_paddings - Prefix paddings
        of shape [source_batch, prefix_len].
      num_hyps_per_beam: An int, number hyps to keep for source sentence.

    Returns:
      initial_results: A `.NestedMap` of initial beam search results.
        log_probs - Log prob for each of the tokens in the target vocab,
                    of shape [target_batch, vocab_size].
        atten_probs - The updated attention probs, of shape
                      [target_batch, source_time].
      states: A `.NestedMap` of initial model states.
        prefix_states - A `.NestedMap` representing prefix decoded states.
        key   - [target_time, target_batch, num_heads, dim_per_head].
        value - [target_time, target_batch, num_heads, dim_per_head].
        time_step - A scalar, the initial decode step set as prefix_len.
    """
    p = self.params
    if p.prefix_len <= 0:
      return super()._InitBeamSearchStateCallback(theta, encoder_outputs,
                                                  num_hyps_per_beam)
    # [source_batch, source_time, dim]
    encoder_out_bm = self._MaybeTransposeEncoderOutputs(encoder_outputs, 'BTC')

    aux_vec = encoder_out_bm.encoded
    aux_paddings = encoder_out_bm.padding

    source_batch = py_utils.GetShape(aux_vec)[0]
    target_batch = source_batch * num_hyps_per_beam
    source_time = py_utils.GetShape(aux_vec)[1]

    log_probs = tf.zeros([target_batch, p.softmax.num_classes],
                         dtype=py_utils.FPropDtype(p))
    # Dummy attention probs
    atten_probs = (
        tf.ones([target_batch, source_time], dtype=py_utils.FPropDtype(p)) /
        tf.cast(source_time, py_utils.FPropDtype(p)))
    initial_results = py_utils.NestedMap(
        log_probs=log_probs, atten_probs=atten_probs)

    prefix_states = py_utils.NestedMap()
    for layer in range(p.num_trans_layers):
      prefix_states['layer_%d' % layer] = self.decoder_trans[layer].InitStates(
          theta.decoder_trans[layer], target_batch,
          p.target_seq_len + p.prefix_len)

    # Always set step_ids for prefix mode.
    initial_results['step_ids'] = tf.expand_dims(
        self._ExpandToNumHyps(encoder_outputs.init_step_ids, num_hyps_per_beam),
        1)

    # Fprop prefix through decoder to init prefix_states.
    # Ref: google3/third_party/py/lingvo/tasks/mt/decoder.py
    prefix_ids = encoder_out_bm.prefix_ids
    prefix_paddings = encoder_out_bm.prefix_paddings

    prefix_ids = tf.tile(input=prefix_ids, multiples=[num_hyps_per_beam, 1])
    prefix_paddings = tf.tile(
        input=prefix_paddings, multiples=[num_hyps_per_beam, 1])

    # Transpose for the layer computation.
    prefix_ids = self._TransposeBeamFormat(prefix_ids, num_hyps_per_beam,
                                           source_batch)

    prefix_paddings = self._TransposeBeamFormat(prefix_paddings,
                                                num_hyps_per_beam, source_batch)

    aux_vec = tf.tile(input=aux_vec, multiples=[num_hyps_per_beam, 1, 1])
    aux_vec = self._TransposeBeamFormat(aux_vec, num_hyps_per_beam,
                                        source_batch)
    aux_paddings = tf.tile(input=aux_paddings, multiples=[num_hyps_per_beam, 1])
    aux_paddings = self._TransposeBeamFormat(aux_paddings, num_hyps_per_beam,
                                             source_batch)

    with tf.name_scope(p.name):
      # [batch, target_time]
      target_ids = prefix_ids
      target_paddings = prefix_paddings
      target_time = py_utils.GetShape(target_ids)[1]

      # Embedding layer
      # [batch, target_time, dim]
      if not p.shared_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, target_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, target_ids)
      # [1, target_time, dim]
      posit_embs = tf.expand_dims(
          self.position_emb.FProp(theta.position_emb, target_time), 0)
      # [batch, target_time, dim]
      input_embs = token_embs + posit_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)
        target_paddings = tf.cast(target_paddings,
                                  p.input_dropout_tpl.fprop_dtype)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      layer_in = input_embs

      dim = py_utils.GetShape(aux_vec)[2]
      layer_in = tf.reshape(layer_in, [target_batch, target_time, dim])
      target_paddings = tf.reshape(target_paddings, [target_batch, target_time])

      if not p.has_aux_atten:
        aux_vec = None
        aux_paddings = None

      for i, (layer, layer_theta) in enumerate(
          zip(self.decoder_trans, theta.decoder_trans)):
        layer_out, _ = layer.FProp(
            layer_theta,
            layer_in,
            target_paddings,
            aux_vec,
            aux_paddings,
            segment_mask=None,
            aux_segment_mask=None)

        # Apply layer_norm if needed.
        if layer.self_atten.params.ln_tpl:
          layer_in = layer.self_atten.layer_norm.FProp(
              layer_theta.self_atten.layer_norm, layer_in)

        for t in range(p.prefix_len):
          query_vec = py_utils.HasShape(layer_in[:, t:t + 1, :],
                                        [target_batch, 1, dim])
          prefix_states = self._UpdatePrefixStates(i, layer, layer_theta,
                                                   query_vec, t, prefix_states)

        layer_in = layer_out

    return initial_results, py_utils.NestedMap({
        'prefix_states': prefix_states,
        'time_step': tf.constant(p.prefix_len)
    })

  def ExtendStep(self,
                 theta,
                 encoder_outputs,
                 new_ids,
                 t,
                 prefix_states,
                 use_short_seq_opt=False):
    """Extend prefix as represented by `prefix_states` by one more step.

    For PrefixDecoder, t starts at position = prefix_len so we need to adjust
    it back to its actual position in the sequence.

    Ref: google3/third_party/py/lingvo/tasks/mt/decoder.py

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: A '.NestedMap' object computed by encoder.
        - encoded: Source encoding of shape [source_time, source_batch, dim] or
          [source_batch, source_time, dim], depending on p.input_data_format.
        - paddings: Source encoding's padding of shape [source_time,
          source_batch] or [source_batch, source_time].
      new_ids: New input ids, of shape [target_batch, 1].
      t: A scalar, the current decode step, 0-based.
      prefix_states: A `.NestedMap` representing the previous decoded states.
        - key: [target_time, target_batch, num_heads, dim_per_head].
        - value: [target_time, target_batch, num_heads, dim_per_head].
      use_short_seq_opt: A bool, whether using short sequence optimization.

    Returns:
      last_decoder_out: The last decoder layer of shape [target_batch, dim].
      updated_prefix_states: A `.NestedMap` representing the updated states.

        - key: [target_time, target_batch, num_heads, dim_per_head].
        - value: [target_time, target_batch, num_heads, dim_per_head].
    """
    p = self.params
    if p.prefix_len <= 0:
      return super().ExtendStep(theta, encoder_outputs, new_ids, t,
                                prefix_states, use_short_seq_opt)

    encoder_out_bm = self._MaybeTransposeEncoderOutputs(encoder_outputs, 'BTC')
    # [source_batch, source_time, dim]
    aux_vec = encoder_out_bm.encoded
    # [source_batch, source_time]
    aux_paddings = encoder_out_bm.padding

    with tf.name_scope(p.name):
      # Embedding layer
      # [target_batch, 1, dim]
      if not p.shared_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, new_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, new_ids)

      num_hyps_per_beam = tf.div(
          py_utils.GetShape(new_ids)[0],
          py_utils.GetShape(aux_paddings)[0])
      t = py_utils.with_dependencies([
          py_utils.assert_between(t, p.prefix_len,
                                  p.prefix_len + p.target_seq_len)
      ], t)
      prefix_len = tf.cast(
          tf.reduce_sum(1 - encoder_out_bm['prefix_paddings'], axis=1), t.dtype)

      # If t >= prefix_len, adjust the t to the actual position.
      adjusted_t = tf.cond(
          t >= p.prefix_len, lambda: t - p.prefix_len + prefix_len - 1,
          lambda: tf.ones_like(prefix_len - 1, dtype=prefix_len.dtype) * t)
      adjusted_t = self._ExpandToNumHyps(adjusted_t, num_hyps_per_beam)
      adjusted_t = self._TransposeBeamFormat(adjusted_t, num_hyps_per_beam,
                                             py_utils.GetShape(aux_paddings)[0])
      adjusted_t_one_hot = tf.one_hot(adjusted_t,
                                      p.prefix_len + p.target_seq_len)
      posit_embs = tf.einsum(
          'ij,jk->ik', adjusted_t_one_hot,
          self.position_emb.FProp(theta.position_emb,
                                  p.prefix_len + p.target_seq_len))
      posit_embs = tf.expand_dims(posit_embs, axis=1)

      # [target_batch, 1, dim]
      input_embs = token_embs + posit_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)

      # Make a copy of the input.
      updated_prefix_states = prefix_states.DeepCopy()

      # Compute the correct mask for prefix. Example: if prefix is [t0 t1 pad]
      # t = 3 and target_seq_len = 3, padding should be [0 1 1 0 1 1].
      # since t1 is used as the first infeed step_id of beam search.
      prefix_len = self._ExpandToNumHyps(prefix_len, num_hyps_per_beam)
      prefix_len = self._TransposeBeamFormat(prefix_len, num_hyps_per_beam,
                                             py_utils.GetShape(aux_paddings)[0])
      last_prefix_mask = tf.one_hot(
          prefix_len - 1,
          py_utils.GetShape(encoder_out_bm['prefix_paddings'])[1],
          dtype=encoder_out_bm['prefix_paddings'].dtype)
      prefix_paddings = tf.tile(
          input=encoder_out_bm['prefix_paddings'],
          multiples=[num_hyps_per_beam, 1])
      prefix_paddings = self._TransposeBeamFormat(
          prefix_paddings, num_hyps_per_beam,
          py_utils.GetShape(aux_paddings)[0]) + last_prefix_mask
      beam_paddings = tf.zeros(
          [py_utils.GetShape(prefix_paddings)[0], t - p.prefix_len + 1],
          dtype=prefix_paddings.dtype)
      per_step_padding = tf.concat([prefix_paddings, beam_paddings], axis=1)

      per_step_padding = py_utils.PadOrTrimTo(
          per_step_padding, [
              py_utils.GetShape(per_step_padding)[0],
              p.prefix_len + p.target_seq_len
          ],
          pad_val=1)
      # [target_batch, 1, target_time]
      per_step_padding = tf.expand_dims(per_step_padding, axis=1)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      layer_in = input_embs

      if not p.has_aux_atten:
        aux_vec = None
        aux_paddings = None

      for i, (layer, layer_theta) in enumerate(
          zip(self.decoder_trans, theta.decoder_trans)):
        # [target_batch, 1, dim]
        layer_out, _, updated_states = layer.ExtendStep(
            layer_theta,
            layer_in,
            aux_vec,
            aux_paddings,
            prefix_states['layer_%i' % i],
            t,
            use_short_seq_opt,
            per_step_padding=per_step_padding)
        updated_prefix_states['layer_%i' % i] = updated_states
        layer_in = layer_out

      # [target_batch, dim]
      last_decoder_out = tf.squeeze(layer_out, 1)
      if p.final_layer_norm:
        last_decoder_out = self.final_ln.FProp(theta.final_ln, last_decoder_out)
      return last_decoder_out, updated_prefix_states

  def SampleTargetSequences(self, theta, encoder_outputs, random_seed):
    """Performs target sequence sampling.

    Ref: google3/third_party/py/lingvo/tasks/mt/decoder.py

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      encoder_outputs: a NestedMap computed by encoder.
      random_seed: a scalar int32 tensor representing the random seed.

    Returns:
      A `.NestedMap` containing output of last decoder layer and attention probs

      - softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
      - attention: `.NestedMap` of attention distributions of shape
        [batch, time, source_len].
    """
    p = self.params
    # TODO(ziruiw): test to see if this line can be removed
    assert self.params.beam_search.num_hyps_per_beam == 1

    # Use init steps in prefix mode, use SOS otherwise.
    init_step_ids = encoder_outputs.init_step_ids if p.prefix_len > 0 else None
    sample = self.target_sequence_sampler.Sample(
        theta, encoder_outputs, random_seed, self._InitBeamSearchStateCallback,
        self._PreBeamSearchStepCallback, self._PostBeamSearchStepCallback,
        init_step_ids)
    bs = tf.shape(sample.ids)[0]
    sample.topk_hyps = tf.zeros([bs, 1], dtype=tf.string)
    sample.topk_ids = sample.ids
    weights = 1 - sample.paddings
    sample.topk_lens = tf.cast(tf.reduce_sum(weights, axis=1), dtype=tf.int32)
    sample.topk_scores = tf.reduce_sum(
        tf.math.log(tf.reduce_max(tf.nn.softmax(sample.logits), axis=2)) *
        weights,
        axis=1)
    return sample

class PrefixTransformerBatchMajorDecoderMixed(decoder.TransformerBatchMajorDecoderMixed):
  """Transformer batch-major decoder that supports prefix inputs and extra embeddings

  It allows four settings by setting prefix_len and has_aux_atten params:
    (1) With Encoder & No Prefix: prefix_len = 0, has_aux_atten = True. This
    corresponds to generic encoder-decoder architectures.
    (2) With Encoder & With Prefix: prefix_len > 0, has_aux_atten = True. This
    corresponds to encoder-decoder architectures with additional prefix in the
    decoder.
    (3) No Encoder & With Prefix: prefix_len > 0, has_aux_atten = False. This
    corresponds to generic prefix LM with decoder-only architecture.
    (4) No Encoder & No Prefix: prefix_len = 0, has_aux_atten = False. This
    corresponds to generic LM with decoder-only architecture.

  In addition, it also supports customized transformer inputs such as images.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('prefix_len', 0,
             ('Max prefix length used in decoding. Disabled if set to 0.'
              'Notice that this is an indicator of prefix mode but this max len'
              'is used for decoding only, the user can set arbitrary prefix'
              'length during training.'))
    p.Define('has_aux_atten', True,
             'Set to enable attention over encoder outputs.')
    return p

  def __init__(self, params):
    if isinstance(params.trans_decoder_tpl, list):
      has_aux_atten = params.trans_decoder_tpl[0].has_aux_atten
    else:
      has_aux_atten = params.trans_decoder_tpl.has_aux_atten
    assert params.has_aux_atten == has_aux_atten
    super().__init__(params)

  def _FProp(self, theta, encoder_outputs, targets):
    p = self.params
    if not p.has_aux_atten:
      # Decoder-only architecture with no encoder atten.
      encoder_outputs = None
    if p.prefix_len > 0 and 'token_visibility' not in targets:
      raise ValueError('Must set token_visibility for prefix in decoder!')
    target_embeddings = self.FPropEmbeddings(theta, targets)
    target_embeddings['token_visibility'] = getattr(targets, 'token_visibility',
                                                    None)
    return self.FPropTransformerLayers(theta, encoder_outputs,
                                       target_embeddings)

  def FPropEmbeddings(self, theta, input_batch):
    """Compute the embeddings for text tokens.

    To enable prefix in the decoder, simply add prefix ids and paddings to
    input_batch and set token_visibility (default to autoregressive).
    For example:
      input_batch.ids = [1, 2, 0, 3, 4, 0]
      input_batch.paddings = [0, 0, 1, 0, 0, 1]
      input_batch.token_visibility = [0, 0, 0, 1, 2, 3]
      Then the first 3 tokens will then be treated as prefix tokens, where the
      position will become: [0, 1, 1, 2, 3, 3].

    Args:
      theta: A NestedMap containing weights' values of this layer and its
        children layers.
      input_batch: a NestedMap containing target input fields.

    Returns:
      A NestedMap containing Transformer input embeddings and paddings.
    """
    p = self.params
    with tf.name_scope(p.name):
      # [batch, target_time]
      target_ids = input_batch.ids
      target_paddings = input_batch.paddings

      # Embedding layer
      # [batch, target_time, dim]
      if not p.shared_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, target_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, target_ids)
      # [batch, target_time, dim]
      # Take care of prefix padding and adjust non-prefix position.
      positions = tf.math.cumsum(
          tf.cast(1. - target_paddings, tf.int32), axis=1) - 1
      posit_embs = self.position_emb.FPropWithPosition(theta.position_emb,
                                                       positions)

      # [batch, target_time, dim]
      input_embs = token_embs + posit_embs
      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)
        target_paddings = tf.cast(target_paddings,
                                  p.input_dropout_tpl.fprop_dtype)

      transformer_input = self.input_dropout.FProp(theta.input_dropout,
                                                   input_embs)
    return py_utils.NestedMap(
        transformer_input=transformer_input, paddings=target_paddings)

  def FPropTransformerLayers(self, theta, encoder_outputs, input_batch):
    """Compute the transformer outputs for input embeddings.

    To enable prefix in the decoder, simply set token_visibility.
    For example:
      input_batch.token_visibility = [0, 0, 1]
      => per_step_padding = [[0, 0, 1], [0, 0, 1], [0, 0, 0]]
      Then the first 2 tokens will then be treated as prefix and are allowed to
      attend to each other but not the last token, while the last one can attend
      to all tokens.

    Args:
      theta: A NestedMap containing weights' values of this layer and its
        children layers.
      encoder_outputs: a NestedMap computed by encoder.
      input_batch: a NestedMap containing target input fields.

    Returns:
      softmax_input: Tensor of shape [target_time, batch, dim].
    """
    p = self.params
    # [batch, target_time, dim]
    layer_in = input_batch.transformer_input
    target_paddings = input_batch.paddings

    per_step_padding = None
    token_visibility = getattr(input_batch, 'token_visibility', None)
    if token_visibility is not None:
      per_step_padding = model_utils.ComputePerStepPadding(
          token_visibility,
          tf.ones_like(token_visibility, dtype=token_visibility.dtype))

    # [batch, source_time, dim]
    aux_vec = None
    aux_paddings = None
    if encoder_outputs is not None:
      if not p.has_aux_atten:
        raise ValueError('Encoder output must be none for decoder only!')
      encoder_out_bm = self._MaybeTransposeEncoderOutputs(
          encoder_outputs, 'BTC')
      aux_vec = encoder_out_bm.encoded
      aux_paddings = encoder_out_bm.padding
      if aux_vec is None or aux_paddings is None:
        raise ValueError('Encoder output must be set for enc-dec architecture!')

    with tf.name_scope(p.name):
      for layer, layer_theta in zip(self.decoder_trans, theta.decoder_trans):
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
            per_step_padding_override=per_step_padding,
            segment_mask=None,
            aux_segment_mask=None)
        layer_in = layer_out

      if p.final_layer_norm:
        layer_out = self.final_ln.FProp(theta.final_ln, layer_out)
      if p.prediction_data_format == 'TBC':
        # Transpose the softmax_input to match the input requirement of
        # ComputePredictions.
        layer_out = tf.transpose(layer_out, [1, 0, 2])
    return layer_out

  def ComputeLoss(self, theta, predictions, targets):
    p = self.params
    if isinstance(predictions, py_utils.NestedMap):
      predictions = predictions.softmax_input
    target_time = py_utils.GetShape(predictions)[{
        'TBC': 0,
        'BTC': 1
    }.get(p.prediction_data_format)]
    batch = py_utils.GetShape(targets.labels)[0]

    # We allow passing labels for non-prefix only and pad here explicitly.
    # Notice that we pad BEFORE original inputs here, corresponding to prefix.
    targets.labels = py_utils.PadOrTrimTo(
        targets.labels, [batch, target_time], pad_after_contents=False)
    targets.weights = py_utils.PadOrTrimTo(
        targets.weights, [batch, target_time], pad_after_contents=False)
    targets.paddings = py_utils.PadOrTrimTo(
        targets.paddings, [batch, target_time], pad_after_contents=False)
    return super().ComputeLoss(theta, predictions, targets)

  def _TransposeBeamFormat(self,
                           tensor,
                           num_hyps_per_beam,
                           num_beam,
                           mode='HB2BH'):
    original_shape = py_utils.GetShape(tensor)

    if mode == 'HB2BH':
      input_shape = [num_hyps_per_beam, num_beam]
    elif mode == 'BH2HB':
      input_shape = [num_beam, num_hyps_per_beam]
    tensor = tf.reshape(tensor, input_shape + original_shape[1:])
    tensor = tf.transpose(tensor,
                          [1, 0] + list(range(2,
                                              len(original_shape) + 1)))
    tensor = tf.reshape(tensor, [-1] + original_shape[1:])
    return tensor

  def _MaybeTransposeField(self, input_map, output_map, field):
    if getattr(input_map, field, None) is None:
      output_map[field] = None
    else:
      output_map[field] = tf.transpose(input_map[field])

  def _MaybeTransposeEncoderOutputs(self, encoder_outputs, target_data_format):
    p = self.params
    if p.input_data_format == target_data_format:
      return encoder_outputs
    transposed = py_utils.NestedMap(
        encoded=tf.transpose(encoder_outputs.encoded, [1, 0, 2]),
        padding=tf.transpose(encoder_outputs.padding))
    self._MaybeTransposeField(encoder_outputs, transposed, 'prefix_ids')
    self._MaybeTransposeField(encoder_outputs, transposed, 'prefix_paddings')
    self._MaybeTransposeField(encoder_outputs, transposed, 'segment_id')
    return transposed

  def AddExtraDecodingInfo(self, encoder_outputs, targets):
    """Adds extra decoding information to encoded_outputs.

    For PrefixDecoder, targets must contain prefix_ids and prefix_paddings
    with shape [bz, prefix_len].

    Args:
      encoder_outputs: a NestedMap computed by encoder.
      targets: a NestedMap containing target input fields.

    Returns:
      encoder_ouputs with prefix ids and paddings to be delivered to decoder.
    """
    p = self.params
    if not p.has_aux_atten:
      if p.prefix_len <= 0:
        batch = py_utils.GetShape(targets.ids)[0]
      else:
        batch = py_utils.GetShape(targets.prefix_ids)[0]
      if p.input_data_format == 'TBC':
        encoded_shape = [1, batch, p.model_dim]
        padding_shape = [1, batch]
      else:
        encoded_shape = [batch, 1, p.model_dim]
        padding_shape = [batch, 1]
      # Dummy encoded and padding. We explicitly set to None for readability,
      # but it is ok to keep dummy values because they will be ignored.
      encoder_outputs = py_utils.NestedMap(
          encoded=tf.zeros(encoded_shape, dtype=tf.int32),
          padding=tf.zeros(padding_shape, dtype=tf.float32))

    if p.prefix_len <= 0:
      return super().AddExtraDecodingInfo(encoder_outputs, targets)

    if not ('prefix_ids' in targets and 'prefix_paddings' in targets):
      raise ValueError('targets must contain prefix_ids and prefix_paddings!')

    # Set the last prefix_ids as init_step_ids, use sos_id if empty.
    prefix_len = tf.cast(
        tf.reduce_sum(1 - targets.prefix_paddings, axis=1),
        targets.prefix_ids.dtype)
    prefix_one_hot = tf.one_hot(
        prefix_len - 1,
        py_utils.GetShape(targets.prefix_ids)[1],
        dtype=targets.prefix_ids.dtype)

    encoder_outputs['init_step_ids'] = tf.reduce_sum(
        prefix_one_hot * targets.prefix_ids, axis=1)

    nonempty_prefix = tf.cast(tf.greater(prefix_len, 0), prefix_one_hot.dtype)
    sos_prefix = tf.ones_like(encoder_outputs.init_step_ids) * p.target_sos_id
    encoder_outputs[
        'init_step_ids'] = nonempty_prefix * encoder_outputs.init_step_ids + (
            1 - nonempty_prefix) * sos_prefix

    # [batch, prefix_len]
    encoder_outputs['prefix_ids'] = py_utils.HasShape(
        targets.prefix_ids,
        [py_utils.GetShape(targets.prefix_ids)[0], p.prefix_len])
    encoder_outputs['prefix_paddings'] = py_utils.HasShape(
        targets.prefix_paddings,
        [py_utils.GetShape(targets.prefix_ids)[0], p.prefix_len])

    # Transpose to be consistent with input format.
    if p.input_data_format == 'TBC':
      encoder_outputs.prefix_ids = tf.transpose(encoder_outputs.prefix_ids)
      encoder_outputs.prefix_paddings = tf.transpose(
          encoder_outputs.prefix_paddings)
    return encoder_outputs

  def _UpdatePrefixStates(self, layer_id, layer, layer_theta, query_vec, t,
                          prefix_states):
    p = self.params

    target_batch = py_utils.GetShape(query_vec)[0]
    attention_dim = p.trans_decoder_tpl.tr_atten_tpl.input_dim
    num_heads = p.trans_decoder_tpl.tr_atten_tpl.num_heads
    per_head_dim = attention_dim // num_heads

    updated_prefix_states = prefix_states.DeepCopy()

    new_key_proj = layer.self_atten.atten.key.FProp(
        layer_theta.self_atten.atten.key, query_vec)
    new_value_proj = layer.self_atten.atten.value.FProp(
        layer_theta.self_atten.atten.value, query_vec)
    new_key_proj = tf.cast(
        tf.reshape(new_key_proj, [target_batch, num_heads, per_head_dim]),
        dtype=prefix_states['layer_%d' % layer_id].key.dtype)
    new_value_proj = tf.cast(
        tf.reshape(new_value_proj, [target_batch, num_heads, per_head_dim]),
        dtype=prefix_states['layer_%d' % layer_id].value.dtype)
    extended_key = scatter_update.Update(
        prefix_states['layer_%d' % layer_id].key, tf.convert_to_tensor(t),
        new_key_proj)
    extended_value = scatter_update.Update(
        prefix_states['layer_%d' % layer_id].value, tf.convert_to_tensor(t),
        new_value_proj)
    updated_prefix_states['layer_%d' % layer_id] = py_utils.NestedMap(
        key=extended_key, value=extended_value)
    return updated_prefix_states

  def _InitBeamSearchStateCallback(self, theta, encoder_outputs,
                                   num_hyps_per_beam):
    """Returns initial beams search states.

    For PrefixDecoder, we FProp prefix sequences to set as the prefix_states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: A '.NestedMap' object computed by encoder. * encoded -
        Source encoding of shape [source_time, source_batch, dim] or
        [source_batch, source_time, dim], depending on p.input_data_format. *
        paddings - Source encoding's padding of shape [source_time,
        source_batch] or [source_batch, source_time]. * prefix_ids - Prefix Ids
        of shape [source_batch, prefix_len]. * prefix_paddings - Prefix paddings
        of shape [source_batch, prefix_len].
      num_hyps_per_beam: An int, number hyps to keep for source sentence.

    Returns:
      initial_results: A `.NestedMap` of initial beam search results.
        log_probs - Log prob for each of the tokens in the target vocab,
                    of shape [target_batch, vocab_size].
        atten_probs - The updated attention probs, of shape
                      [target_batch, source_time].
      states: A `.NestedMap` of initial model states.
        prefix_states - A `.NestedMap` representing prefix decoded states.
        key   - [target_time, target_batch, num_heads, dim_per_head].
        value - [target_time, target_batch, num_heads, dim_per_head].
        time_step - A scalar, the initial decode step set as prefix_len.
    """
    p = self.params
    if p.prefix_len <= 0:
      return super()._InitBeamSearchStateCallback(theta, encoder_outputs,
                                                  num_hyps_per_beam)
    # [source_batch, source_time, dim]
    encoder_out_bm = self._MaybeTransposeEncoderOutputs(encoder_outputs, 'BTC')

    aux_vec = encoder_out_bm.encoded
    aux_paddings = encoder_out_bm.padding

    source_batch = py_utils.GetShape(aux_vec)[0]
    target_batch = source_batch * num_hyps_per_beam
    source_time = py_utils.GetShape(aux_vec)[1]

    log_probs = tf.zeros([target_batch, p.softmax.num_classes],
                         dtype=py_utils.FPropDtype(p))
    # Dummy attention probs
    atten_probs = (
        tf.ones([target_batch, source_time], dtype=py_utils.FPropDtype(p)) /
        tf.cast(source_time, py_utils.FPropDtype(p)))
    initial_results = py_utils.NestedMap(
        log_probs=log_probs, atten_probs=atten_probs)

    prefix_states = py_utils.NestedMap()
    for layer in range(p.num_trans_layers):
      prefix_states['layer_%d' % layer] = self.decoder_trans[layer].InitStates(
          theta.decoder_trans[layer], target_batch,
          p.target_seq_len + p.prefix_len)

    # Always set step_ids for prefix mode.
    initial_results['step_ids'] = tf.expand_dims(
        self._ExpandToNumHyps(encoder_outputs.init_step_ids, num_hyps_per_beam),
        1)

    # Fprop prefix through decoder to init prefix_states.
    # Ref: google3/third_party/py/lingvo/tasks/mt/decoder.py
    prefix_ids = encoder_out_bm.prefix_ids
    prefix_paddings = encoder_out_bm.prefix_paddings

    prefix_ids = tf.tile(input=prefix_ids, multiples=[num_hyps_per_beam, 1])
    prefix_paddings = tf.tile(
        input=prefix_paddings, multiples=[num_hyps_per_beam, 1])

    # Transpose for the layer computation.
    prefix_ids = self._TransposeBeamFormat(prefix_ids, num_hyps_per_beam,
                                           source_batch)

    prefix_paddings = self._TransposeBeamFormat(prefix_paddings,
                                                num_hyps_per_beam, source_batch)

    aux_vec = tf.tile(input=aux_vec, multiples=[num_hyps_per_beam, 1, 1])
    aux_vec = self._TransposeBeamFormat(aux_vec, num_hyps_per_beam,
                                        source_batch)
    aux_paddings = tf.tile(input=aux_paddings, multiples=[num_hyps_per_beam, 1])
    aux_paddings = self._TransposeBeamFormat(aux_paddings, num_hyps_per_beam,
                                             source_batch)

    with tf.name_scope(p.name):
      # [batch, target_time]
      target_ids = prefix_ids
      target_paddings = prefix_paddings
      target_time = py_utils.GetShape(target_ids)[1]

      # Embedding layer
      # [batch, target_time, dim]
      if not p.shared_emb and not p.shared_emb_ex:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, target_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, target_ids)
      # [1, target_time, dim]
      posit_embs = tf.expand_dims(
          self.position_emb.FProp(theta.position_emb, target_time), 0)
      # [batch, target_time, dim]
      input_embs = token_embs + posit_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)
        target_paddings = tf.cast(target_paddings,
                                  p.input_dropout_tpl.fprop_dtype)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      layer_in = input_embs

      dim = py_utils.GetShape(aux_vec)[2]
      layer_in = tf.reshape(layer_in, [target_batch, target_time, dim])
      target_paddings = tf.reshape(target_paddings, [target_batch, target_time])

      if not p.has_aux_atten:
        aux_vec = None
        aux_paddings = None

      for i, (layer, layer_theta) in enumerate(
          zip(self.decoder_trans, theta.decoder_trans)):
        layer_out, _ = layer.FProp(
            layer_theta,
            layer_in,
            target_paddings,
            aux_vec,
            aux_paddings,
            segment_mask=None,
            aux_segment_mask=None)

        # Apply layer_norm if needed.
        if layer.self_atten.params.ln_tpl:
          layer_in = layer.self_atten.layer_norm.FProp(
              layer_theta.self_atten.layer_norm, layer_in)

        for t in range(p.prefix_len):
          query_vec = py_utils.HasShape(layer_in[:, t:t + 1, :],
                                        [target_batch, 1, dim])
          prefix_states = self._UpdatePrefixStates(i, layer, layer_theta,
                                                   query_vec, t, prefix_states)

        layer_in = layer_out

    return initial_results, py_utils.NestedMap({
        'prefix_states': prefix_states,
        'time_step': tf.constant(p.prefix_len)
    })

  def ExtendStep(self,
                 theta,
                 encoder_outputs,
                 new_ids,
                 t,
                 prefix_states,
                 use_short_seq_opt=False):
    """Extend prefix as represented by `prefix_states` by one more step.

    For PrefixDecoder, t starts at position = prefix_len so we need to adjust
    it back to its actual position in the sequence.

    Ref: google3/third_party/py/lingvo/tasks/mt/decoder.py

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: A '.NestedMap' object computed by encoder.
        - encoded: Source encoding of shape [source_time, source_batch, dim] or
          [source_batch, source_time, dim], depending on p.input_data_format.
        - paddings: Source encoding's padding of shape [source_time,
          source_batch] or [source_batch, source_time].
      new_ids: New input ids, of shape [target_batch, 1].
      t: A scalar, the current decode step, 0-based.
      prefix_states: A `.NestedMap` representing the previous decoded states.
        - key: [target_time, target_batch, num_heads, dim_per_head].
        - value: [target_time, target_batch, num_heads, dim_per_head].
      use_short_seq_opt: A bool, whether using short sequence optimization.

    Returns:
      last_decoder_out: The last decoder layer of shape [target_batch, dim].
      updated_prefix_states: A `.NestedMap` representing the updated states.

        - key: [target_time, target_batch, num_heads, dim_per_head].
        - value: [target_time, target_batch, num_heads, dim_per_head].
    """
    p = self.params
    if p.prefix_len <= 0:
      return super().ExtendStep(theta, encoder_outputs, new_ids, t,
                                prefix_states, use_short_seq_opt)

    encoder_out_bm = self._MaybeTransposeEncoderOutputs(encoder_outputs, 'BTC')
    # [source_batch, source_time, dim]
    aux_vec = encoder_out_bm.encoded
    # [source_batch, source_time]
    aux_paddings = encoder_out_bm.padding

    with tf.name_scope(p.name):
      # Embedding layer
      # [target_batch, 1, dim]
      if not p.shared_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, new_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, new_ids)

      num_hyps_per_beam = tf.div(
          py_utils.GetShape(new_ids)[0],
          py_utils.GetShape(aux_paddings)[0])
      t = py_utils.with_dependencies([
          py_utils.assert_between(t, p.prefix_len,
                                  p.prefix_len + p.target_seq_len)
      ], t)
      prefix_len = tf.cast(
          tf.reduce_sum(1 - encoder_out_bm['prefix_paddings'], axis=1), t.dtype)

      # If t >= prefix_len, adjust the t to the actual position.
      adjusted_t = tf.cond(
          t >= p.prefix_len, lambda: t - p.prefix_len + prefix_len - 1,
          lambda: tf.ones_like(prefix_len - 1, dtype=prefix_len.dtype) * t)
      adjusted_t = self._ExpandToNumHyps(adjusted_t, num_hyps_per_beam)
      adjusted_t = self._TransposeBeamFormat(adjusted_t, num_hyps_per_beam,
                                             py_utils.GetShape(aux_paddings)[0])
      adjusted_t_one_hot = tf.one_hot(adjusted_t,
                                      p.prefix_len + p.target_seq_len)
      posit_embs = tf.einsum(
          'ij,jk->ik', adjusted_t_one_hot,
          self.position_emb.FProp(theta.position_emb,
                                  p.prefix_len + p.target_seq_len))
      posit_embs = tf.expand_dims(posit_embs, axis=1)

      # [target_batch, 1, dim]
      input_embs = token_embs + posit_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)

      # Make a copy of the input.
      updated_prefix_states = prefix_states.DeepCopy()

      # Compute the correct mask for prefix. Example: if prefix is [t0 t1 pad]
      # t = 3 and target_seq_len = 3, padding should be [0 1 1 0 1 1].
      # since t1 is used as the first infeed step_id of beam search.
      prefix_len = self._ExpandToNumHyps(prefix_len, num_hyps_per_beam)
      prefix_len = self._TransposeBeamFormat(prefix_len, num_hyps_per_beam,
                                             py_utils.GetShape(aux_paddings)[0])
      last_prefix_mask = tf.one_hot(
          prefix_len - 1,
          py_utils.GetShape(encoder_out_bm['prefix_paddings'])[1],
          dtype=encoder_out_bm['prefix_paddings'].dtype)
      prefix_paddings = tf.tile(
          input=encoder_out_bm['prefix_paddings'],
          multiples=[num_hyps_per_beam, 1])
      prefix_paddings = self._TransposeBeamFormat(
          prefix_paddings, num_hyps_per_beam,
          py_utils.GetShape(aux_paddings)[0]) + last_prefix_mask
      beam_paddings = tf.zeros(
          [py_utils.GetShape(prefix_paddings)[0], t - p.prefix_len + 1],
          dtype=prefix_paddings.dtype)
      per_step_padding = tf.concat([prefix_paddings, beam_paddings], axis=1)

      per_step_padding = py_utils.PadOrTrimTo(
          per_step_padding, [
              py_utils.GetShape(per_step_padding)[0],
              p.prefix_len + p.target_seq_len
          ],
          pad_val=1)
      # [target_batch, 1, target_time]
      per_step_padding = tf.expand_dims(per_step_padding, axis=1)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      layer_in = input_embs

      if not p.has_aux_atten:
        aux_vec = None
        aux_paddings = None

      for i, (layer, layer_theta) in enumerate(
          zip(self.decoder_trans, theta.decoder_trans)):
        # [target_batch, 1, dim]
        layer_out, _, updated_states = layer.ExtendStep(
            layer_theta,
            layer_in,
            aux_vec,
            aux_paddings,
            prefix_states['layer_%i' % i],
            t,
            use_short_seq_opt,
            per_step_padding=per_step_padding)
        updated_prefix_states['layer_%i' % i] = updated_states
        layer_in = layer_out

      # [target_batch, dim]
      last_decoder_out = tf.squeeze(layer_out, 1)
      if p.final_layer_norm:
        last_decoder_out = self.final_ln.FProp(theta.final_ln, last_decoder_out)
      return last_decoder_out, updated_prefix_states

  def SampleTargetSequences(self, theta, encoder_outputs, random_seed):
    """Performs target sequence sampling.

    Ref: google3/third_party/py/lingvo/tasks/mt/decoder.py

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      encoder_outputs: a NestedMap computed by encoder.
      random_seed: a scalar int32 tensor representing the random seed.

    Returns:
      A `.NestedMap` containing output of last decoder layer and attention probs

      - softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
      - attention: `.NestedMap` of attention distributions of shape
        [batch, time, source_len].
    """
    p = self.params
    assert self.params.beam_search.num_hyps_per_beam == 1

    # Use init steps in prefix mode, use SOS otherwise.
    init_step_ids = encoder_outputs.init_step_ids if p.prefix_len > 0 else None
    sample = self.target_sequence_sampler.Sample(
        theta, encoder_outputs, random_seed, self._InitBeamSearchStateCallback,
        self._PreBeamSearchStepCallback, self._PostBeamSearchStepCallback,
        init_step_ids)
    bs = tf.shape(sample.ids)[0]
    sample.topk_hyps = tf.zeros([bs, 1], dtype=tf.string)
    sample.topk_ids = sample.ids
    weights = 1 - sample.paddings
    sample.topk_lens = tf.cast(tf.reduce_sum(weights, axis=1), dtype=tf.int32)
    sample.topk_scores = tf.reduce_sum(
        tf.math.log(tf.reduce_max(tf.nn.softmax(sample.logits), axis=2)) *
        weights,
        axis=1)
    return sample
