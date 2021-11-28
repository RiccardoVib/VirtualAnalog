import pandas as pd
import numpy as np
import copy
import tensorflow as tf
from transformers.models import longformer
from transformers.models.longformer.modeling_tf_longformer import *
from transformers.modeling_tf_utils import keras_serializable, input_processing
from transformers.models.longformer import LongformerConfig
from Code.Models.Transformer import positional_encoding

# TODO REMOVE!!!
from Code.TrainFunctionality import CustomSchedule, get_batches
from scipy.io import wavfile
from scipy import signal
from tensorflow.keras.utils import Progbar
from Code.Preprocess import my_scaler_stft
import matplotlib.pyplot as plt


@keras_serializable
class My_TFLongformerMainLayer(tf.keras.layers.Layer):
    config_class = LongformerConfig

    def __init__(self, config,  **kwargs):
        super().__init__(**kwargs)

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.pad_token_id = config.pad_token_id
        self.attention_window = config.attention_window

        # self.embeddings = TFLongformerEmbeddings(config, name="embeddings")
        self.embedding2 = tf.keras.layers.Dense(config.hidden_size)
        self.embedding3 = tf.keras.layers.Dense(config.hidden_size)
        self.hidden_size = config.hidden_size
        self.pos_encoding = positional_encoding(config.max_position_embeddings, self.hidden_size)      # TODO Make sure to pass this as input to config.

        self.encoder = TFLongformerEncoder(config, name="encoder")
        self.decoder = TFLongformerDecoder(config, name="decoder")
        # self.pooler = TFLongformerPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        attention_mask_dec=None,
        head_mask=None,
        head_mask_dec=None,
        global_attention_mask=None,
        global_attention_mask_dec=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        inputs_embeds_dec=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs_embeds = self.embedding2(inputs_embeds)      # TODO: Also add positional encoding
        inputs_embeds_dec = self.embedding3(inputs_embeds_dec)
        inputs_embeds += self.pos_encoding[:, :tf.shape(inputs_embeds)[1], :]
        inputs_embeds_dec += self.pos_encoding[:, :tf.shape(inputs_embeds_dec)[1], :]


        output_length_dec = tf.shape(inputs_embeds_dec)[1]
        # inputs = input_processing(
        #     func=self.call,
        #     config=self.config,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     head_mask=head_mask,
        #     global_attention_mask=global_attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     training=training,
        #     kwargs_call=kwargs,
        # )

        inputs = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'head_mask': head_mask,
                  'global_attention_mask': global_attention_mask,
                  'token_type_ids': token_type_ids,
                  'position_ids': position_ids,
                  'inputs_embeds': inputs_embeds,
                  'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states,
                  'return_dict': return_dict,
                  'training': training}

        inputs_dec = {'input_ids': input_ids,
                      'attention_mask': attention_mask_dec,
                      'head_mask': head_mask_dec,
                      'global_attention_mask': global_attention_mask_dec,
                      'token_type_ids': token_type_ids,
                      'position_ids': position_ids,
                      'inputs_embeds': inputs_embeds_dec,
                      'output_attentions': output_attentions,
                      'output_hidden_states': output_hidden_states,
                      'return_dict': return_dict,
                      'training': training}

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None and inputs_dec['inputs_embeds'] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
            input_shape_dec = shape_list(inputs_dec['inputs_embeds'])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_dec["attention_mask"] is None:
            inputs_dec["attention_mask"] = tf.fill(input_shape_dec, 1)

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(input_shape, 1)

        # if tf.shape(inputs_dec['attention_mask'])[1] > input_shape_dec[1]:
        #     inputs_dec['attention_mask'] = inputs_dec['attention_mask'][:, :input_shape_dec[1]]

        inputs_dec['attention_mask'] = tf.cond(
            tf.math.greater(tf.shape(inputs_dec['attention_mask'])[1], input_shape_dec[1]),
            lambda: inputs_dec['attention_mask'][:, :input_shape_dec[1]],
            lambda: inputs_dec['attention_mask'])

        if inputs["token_type_ids"] is None:
            inputs["token_type_ids"] = tf.fill(input_shape, 0)

        if inputs_dec["token_type_ids"] is None:
            inputs_dec["token_type_ids"] = tf.fill(input_shape_dec, 0)

        # merge `global_attention_mask` and `attention_mask`
        if inputs["global_attention_mask"] is not None:
            inputs["attention_mask"] = self._merge_to_attention_mask(
                inputs["attention_mask"], inputs["global_attention_mask"]
            )

        # merge `global_attention_mask` and `attention_mask`
        if inputs_dec["global_attention_mask"] is not None:
            inputs_dec['global_attention_mask'] = tf.cond(
                tf.math.greater(tf.shape(inputs_dec['global_attention_mask'])[1], input_shape_dec[1]),
                lambda: inputs_dec['global_attention_mask'][:, :input_shape_dec[1]],
                lambda: inputs_dec['global_attention_mask']
            )
            # if tf.shape(inputs_dec['global_attention_mask'])[1] > input_shape_dec[1]:
            #     inputs_dec['global_attention_mask'] = inputs_dec['global_attention_mask'][:, :input_shape_dec[1]]

            inputs_dec["attention_mask"] = self._merge_to_attention_mask(
                inputs_dec["attention_mask"], inputs_dec["global_attention_mask"]
            )

        # Since we're passing input embeddings, this only pads the input to work with the specified window length.
        # Padded indices will get a 0 in attention mask, while global indices will get 2 and other get 1
        (
            padding_len,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
            inputs["position_ids"],
            inputs["inputs_embeds"],
        ) = self._pad_to_window_size(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            inputs_embeds=inputs["inputs_embeds"],
            pad_token_id=self.pad_token_id,
            pad_to_len=None
        )
        (
            padding_len_dec,
            inputs_dec["input_ids"],
            inputs_dec["attention_mask"],
            inputs_dec["token_type_ids"],
            inputs_dec["position_ids"],
            inputs_dec["inputs_embeds"],
        ) = self._pad_to_window_size(
            input_ids=inputs_dec["input_ids"],
            attention_mask=inputs_dec["attention_mask"],
            token_type_ids=inputs_dec["token_type_ids"],
            position_ids=inputs_dec["position_ids"],
            inputs_embeds=inputs_dec["inputs_embeds"],
            pad_token_id=self.pad_token_id,
            pad_to_len=shape_list(inputs['inputs_embeds'])[1]
        )

        # is index masked or global attention
        is_index_masked = tf.math.less(inputs["attention_mask"], 1)          # True for mask (pad) locations
        is_index_masked_dec = tf.math.less(inputs_dec["attention_mask"], 1)          # True for mask (pad) locations
        is_index_global_attn = tf.math.greater(inputs["attention_mask"], 1)  # True for global locations
        is_index_global_attn_dec = tf.math.greater(inputs_dec["attention_mask"], 1)  # True for global locations
        is_global_attn = tf.math.reduce_any(is_index_global_attn)            # True if any in previous
        is_global_attn_dec = tf.math.reduce_any(is_index_global_attn_dec)            # True if any in previous

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, to_seq_length, 1, 1]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask_shape = shape_list(inputs["attention_mask"])
        attention_mask_shape_dec = shape_list(inputs_dec["attention_mask"])
        extended_attention_mask = tf.reshape(
            inputs["attention_mask"], (attention_mask_shape[0], attention_mask_shape[1], 1, 1)
        )
        extended_attention_mask_dec = tf.reshape(
            inputs_dec["attention_mask"], (attention_mask_shape_dec[0], attention_mask_shape_dec[1], 1, 1)
        )

        # Since attention_mask is 1.0 for positions we want to attend locally and 0.0 for
        # masked and global attn positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(tf.math.abs(1 - extended_attention_mask), tf.dtypes.float32) * -10000.0
        extended_attention_mask_dec = tf.cast(tf.math.abs(1 - extended_attention_mask_dec), tf.dtypes.float32) * -10000.0
        # embedding_output = inputs['x']
        # embedding_output *= tf.math.sqrt(tf.cast(self.hidden_size, tf.float32))  # TODO find out what this is doing
        # seq_len = tf.shape(embedding_output)[1]
        # embedding_output += self.pos_encoding[:, :seq_len, :]

        # embedding_output = self.embeddings(
        #     inputs["input_ids"],
        #     inputs["position_ids"],
        #     inputs["token_type_ids"],
        #     inputs["inputs_embeds"],
        #     training=inputs["training"],
        # )
        embedding_output = inputs['inputs_embeds']
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            padding_len=padding_len,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output_enc = encoder_outputs[0]
        embedding_output_dec = inputs_dec['inputs_embeds']
        decoder_outputs = self.decoder(
            embedding_output_dec,   # TODO
            sequence_output_enc,
            attention_mask1=extended_attention_mask_dec,
            attention_mask2=extended_attention_mask,
            head_mask1=head_mask_dec,
            head_mask2=head_mask,
            padding_len=padding_len_dec,
            is_index_masked=is_index_masked_dec,
            is_index_masked_enc=is_index_masked,
            is_index_global_attn1=is_index_global_attn_dec,
            is_index_global_attn2=is_index_global_attn,
            is_global_attn=is_global_attn_dec,
            output_attentions=inputs_dec["output_attentions"],
            output_hidden_states=inputs_dec["output_hidden_states"],
            return_dict=inputs_dec["return_dict"],
            training=inputs_dec["training"],
            output_length=output_length_dec,
        )
        sequence_output_dec = decoder_outputs[0]
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        pooled_output = None

        # # undo padding
        # if padding_len > 0:
        #     # unpad `sequence_output` because the calling function is expecting a length == input_ids.size(1)
        #     sequence_output_enc = sequence_output_enc[:, :-padding_len]

        # if padding_len_dec > 0:
        #     # unpad `sequence_output` because the calling function is expecting a length == input_ids.size(1)
        #     sequence_output_dec = sequence_output_dec[:, :-padding_len_dec]
        sequence_output_dec = tf.cond(tf.math.greater(padding_len_dec, 0),
                                      lambda: sequence_output_dec[:, :-padding_len_dec],
                                      lambda: sequence_output_dec)

        if not inputs["return_dict"]:
            return (
                sequence_output_dec,
                pooled_output,
            ) + encoder_outputs[1:] + decoder_outputs[1:]      # TODO

        # TODO: Now we need to remove all the token inputs and etc and only feed in the inputs embeds as this will tidy
        # everything up a little bit...
        return TFLongformerEncDecOutput(
            last_hidden_state=sequence_output_dec,
            pooler_output=pooled_output,
            hidden_states_dec=decoder_outputs.hidden_states,
            hidden_states_enc=encoder_outputs.hidden_states,
            attentions_dec=decoder_outputs.attentions,
            attentions_enc=encoder_outputs.attentions,
            global_attentions_dec=decoder_outputs.global_attentions,
            global_attentions_enc=encoder_outputs.global_attentions
        )

    def _pad_to_window_size(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        inputs_embeds,
        pad_token_id,
        pad_to_len
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""
        # padding
        attention_window = (
            self.attention_window if isinstance(self.attention_window, int) else max(self.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"

        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)
        batch_size, seq_len = input_shape[:2]
        if pad_to_len is None:
            padding_len = (attention_window - seq_len % attention_window) % attention_window
        else:
            padding_len = pad_to_len - seq_len

        # if tf.padding_len > 0:
        #     logger.info(
        #         f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
        #         f"`config.attention_window`: {attention_window}"
        #     )
        # tf.cond(tf.math.greater(padding_len, 0),
        #         lambda: logger.info(
        #         f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
        #         f"`config.attention_window`: {attention_window}"))

        paddings = tf.convert_to_tensor([[0, 0], [0, padding_len]])

        if input_ids is not None:
            input_ids = tf.pad(input_ids, paddings, constant_values=pad_token_id)

        if position_ids is not None:
            # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
            position_ids = tf.pad(position_ids, paddings, constant_values=pad_token_id)

        if inputs_embeds is not None:

            def pad_embeddings():
                input_ids_padding = tf.fill((batch_size, padding_len), self.pad_token_id)
                # inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds_padding = tf.zeros(shape=(tf.shape(input_ids_padding)[0],
                                                        tf.shape(input_ids_padding)[1],
                                                        self.hidden_size))
                return tf.concat([inputs_embeds, inputs_embeds_padding], axis=-2)

            inputs_embeds = tf.cond(tf.math.greater(padding_len, 0), pad_embeddings, lambda: inputs_embeds)

        attention_mask = tf.pad(attention_mask, paddings, constant_values=False)  # no attention on the padding tokens
        token_type_ids = tf.pad(token_type_ids, paddings, constant_values=0)  # pad with token_type_id = 0

        return (
            padding_len,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
        )

    @staticmethod
    def _merge_to_attention_mask(attention_mask: tf.Tensor, global_attention_mask: tf.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1

        return attention_mask


@add_start_docstrings(
    "The bare Longformer Model outputting raw hidden-states without any specific head on top.",
    LONGFORMER_START_DOCSTRING,
)
class My_TFLongformerModel(TFLongformerPreTrainedModel):
    """
    This class copies code from :class:`~transformers.TFRobertaModel` and overwrites standard self-attention with
    longformer self-attention to provide the ability to process long sequences following the self-attention approach
    described in `Longformer: the Long-Document Transformer <https://arxiv.org/abs/2004.05150>`__ by Iz Beltagy,
    Matthew E. Peters, and Arman Cohan. Longformer self-attention combines a local (sliding window) and global
    attention to extend to long documents without the O(n^2) increase in memory and compute.
    The self-attention module :obj:`TFLongformerSelfAttention` implemented here supports the combination of local and
    global attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and
    dilated attention are more relevant for autoregressive language modeling than finetuning on downstream tasks.
    Future release will add support for autoregressive attention, but the support for dilated attention requires a
    custom CUDA kernel to be memory and compute efficient.
    """

    def __init__(self, config, output_dim, act_fn_output='tanh', *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.longformer = My_TFLongformerMainLayer(config, name="longformer")
        self.dense_output = tf.keras.layers.Dense(output_dim, activation=act_fn_output, name='DensOutput')

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        # input_ids=None,
        # attention_mask=None,
        # attention_mask_dec=None,
        # head_mask=None,
        # head_mask_dec=None,
        inputs_embeds=None,
        inputs_embeds_dec=None,
        global_attention_mask=None,
        global_attention_mask_dec=None,
        # token_type_ids=None,
        # position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        # return_dict=None,
        training=False,
        **kwargs,
    ):
        # global_attention_mask, global_attention_mask_dec, inputs_embeds, inputs_embeds_dec, output_attentions, output_hidden_states, training = global_attention_mask
        # global_attention_mask = inputs['global_attention_mask'],
        # global_attention_mask_dec = inputs['global_attention_mask_dec'],
        # # token_type_ids=None,
        # # position_ids=None,
        # inputs_embeds = inputs['inputs_embeds'],
        # inputs_embeds_dec = inputs['inputs_embeds_dec'],
        # output_attentions = inputs['output_attentions'],
        # output_hidden_states = inputs['output_hidden_states'],
        # # return_dict=None,
        # training = inputs['training']
        outputs = self.longformer(
            input_ids=None,
            attention_mask=None,
            attention_mask_dec=None,
            head_mask=None,
            head_mask_dec=None,
            global_attention_mask=global_attention_mask,
            global_attention_mask_dec=global_attention_mask_dec,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=inputs_embeds,
            inputs_embeds_dec=inputs_embeds_dec,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            training=training,
        )
        outputs['output'] = self.dense_output(outputs['last_hidden_state'])

        return outputs  # ['output']

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
        g_attns = tf.convert_to_tensor(output.global_attentions) if self.config.output_attentions else None

        return TFLongformerBaseModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=hs,
            attentions=attns,
            global_attentions=g_attns,
        )


if __name__ == '__main__':
    # x = pd.read_pickle(r'C:\Users\larsbent\Downloads\longformTEST.pickle')
    # x = x[:2, :, :]
    # x = tf.constant(x, dtype='float32')

    # x = np.concatenate(
    #     [np.reshape(np.concatenate([np.expand_dims(np.sin(np.deg2rad(np.arange(0, 180, 10))), 1),
    #                                 np.expand_dims(np.cos(np.deg2rad(np.arange(0, 180, 10))), 1)], axis=-1),
    #                 (1, 18, 2)),
    #      np.reshape(np.concatenate([np.expand_dims(np.sin(np.deg2rad(np.arange(90, 270, 10))), 1),
    #                                 np.expand_dims(np.cos(np.deg2rad(np.arange(90, 270, 10))), 1)], axis=-1),
    #                 (1, 18, 2)),
    #      np.reshape(np.concatenate([np.expand_dims(np.sin(np.deg2rad(np.arange(270, 360 + 90, 10))), 1),
    #                                 np.expand_dims(np.cos(np.deg2rad(np.arange(270, 360 + 90, 10))), 1)], axis=-1),
    #                 (1, 18, 2)),
    #      ], axis=0)
    # x_test = np.concatenate(
    #     [np.reshape(np.concatenate([np.expand_dims(np.sin(np.deg2rad(np.arange(30, 210, 10))), 1),
    #                                 np.expand_dims(np.cos(np.deg2rad(np.arange(30, 210, 10))), 1)], axis=-1),
    #                 (1, 18, 2)),
    #      np.reshape(np.concatenate([np.expand_dims(np.sin(np.deg2rad(np.arange(80, 260, 10))), 1),
    #                                 np.expand_dims(np.cos(np.deg2rad(np.arange(80, 260, 10))), 1)], axis=-1),
    #                 (1, 18, 2)),
    #      ], axis=0)
    #
    # x = tf.constant(x, dtype='float32')
    # x_test = tf.constant(x_test, dtype='float32')
    #
    # max_position_embedding = x.shape[1]
    # hidden_size = 12
    # configuration = LongformerConfig(attention_window=[4, 4, 2],
    #                                  pad_token_id=1,        # Pad token id (don't change)
    #                                  hidden_size=hidden_size,
    #                                  num_hidden_layers=3,
    #                                  num_attention_heads=2,
    #                                  intermediate_size=3072,    # For intermediate dense layer
    #                                  hidden_act="tanh",
    #                                  hidden_dropout_prob=0.,
    #                                  attention_probs_dropout_prob=0.,
    #                                  max_position_embeddings=max_position_embedding,    # For positional encoding
    #                                  initializer_range=0.02,
    #                                  layer_norm_eps=1e-12)
    #
    # longformer = My_TFLongformerModel(configuration, output_dim=1, act_fn_output='tanh')
    # np.random.seed(4)
    # global_attention_mask = np.random.choice([0, 1], (1, tf.shape(x)[1]), p=[.9, .1])
    # global_attention_mask[0, -1] = 1
    # global_attention_mask = tf.repeat(global_attention_mask, tf.shape(x)[0], axis=0)
    #
    #
    # learning_rate = CustomSchedule(d_model=hidden_size, warmup_steps=4000)
    # opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, name='MyOpt')
    #
    # loss_fn = tf.keras.losses.MeanAbsoluteError()
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # val_loss = tf.keras.metrics.Mean(name='val_loss')
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    #
    # longformer.compile(optimizer=opt, loss=loss_fn,)
    # input_train = {'inputs_embeds': x[:,:,:1],
    #                'inputs_embeds_dec': x[:, :-1, 1:],
    #                'global_attention_mask': global_attention_mask,
    #                'global_attention_mask_dec': global_attention_mask,
    #                'output_attentions': tf.constant(False, shape=(tf.shape(x)[0],1)),
    #                'output_hidden_states': tf.constant(False, shape=(tf.shape(x)[0],1)),
    #                'training': tf.constant(True, shape=(tf.shape(x)[0],1))}
    # input_val = {'inputs_embeds': x_test[:,:,:1],
    #                'inputs_embeds_dec': x_test[:, :-1, 1:],
    #                'global_attention_mask': global_attention_mask[:2,:],
    #                'global_attention_mask_dec': global_attention_mask[:2,:],
    #                'output_attentions': tf.constant(False, shape=(tf.shape(x_test)[0],1)),
    #                'output_hidden_states': tf.constant(False, shape=(tf.shape(x_test)[0],1)),
    #                'training': tf.constant(False, shape=(tf.shape(x_test)[0],1))}
    #
    # dataset_train_inp = tf.data.Dataset.from_tensor_slices((global_attention_mask, global_attention_mask,
    #                                                     x[:,:,:1], x[:,:-1,1:],
    #                                                     tf.constant(False, shape=(tf.shape(x)[0],1)),
    #                                                     tf.constant(False, shape=(tf.shape(x)[0],1)),
    #                                                     tf.constant(True, shape=(tf.shape(x)[0],1))))
    # dataset_train_lab = tf.data.Dataset.from_tensor_slices(x[:, 1:, 1:])
    # dataset_val_inp = tf.data.Dataset.from_tensor_slices((global_attention_mask[:2,:], global_attention_mask[:2,:],
    #                                                     x_test[:,:,:1], x_test[:,:-1,1:],
    #                                                     tf.constant(False, shape=(tf.shape(x_test)[0],1)),
    #                                                     tf.constant(False, shape=(tf.shape(x_test)[0],1)),
    #                                                     tf.constant(False, shape=(tf.shape(x_test)[0],1))))
    # dataset_val_lab = tf.data.Dataset.from_tensor_slices(x_test[:,1:,1:])
    # dataset_train = tf.data.Dataset.zip((dataset_train_inp, dataset_train_lab)).batch(1).repeat()
    # dataset_val = tf.data.Dataset.zip((dataset_val_inp, dataset_val_lab)).batch(1)
    #
    # #[None, None, None, None, None, global_attention_mask, global_attention_mask, None,None,x[:,:,:1],  x[:, :-1, 1:, False, False,True]
    # longformer.fit(dataset_train,
    #                validation_data=dataset_val,
    #                epochs=1000,
    #                steps_per_epoch=3)

    # # @tf.function
    # def train_step(inp, tar, global_attention_mask=None):
    #     tar_inp = tar[:, :-1, :]
    #     tar_real = tar[:, 1:, :]
    #
    #     with tf.GradientTape() as tape:
    #         predictions = longformer(inputs_embeds=inp,
    #                                  inputs_embeds_dec=tar_inp,
    #                                  global_attention_mask=global_attention_mask,
    #                                  global_attention_mask_dec=global_attention_mask,
    #                                  output_attentions=False,
    #                                  output_hidden_states=False,
    #                                  training=True)
    #         predictions = predictions['output']
    #         loss = loss_fn(tar_real, predictions)
    #     gradients = tape.gradient(loss, longformer.trainable_variables)
    #     opt.apply_gradients(zip(gradients, longformer.trainable_variables))
    #     train_loss.update_state(loss)
    #
    # # @tf.function
    # def val_step(inp, tar, testing=False, global_attention_mask=None):
    #     tar_inp = tar[:, :-1, :]
    #     tar_real = tar[:, 1:, :]
    #
    #     outputs = longformer(inputs_embeds=inp,
    #                          inputs_embeds_dec=tar_inp,
    #                          global_attention_mask=global_attention_mask,
    #                          global_attention_mask_dec=global_attention_mask,
    #                          output_attentions=False,
    #                          output_hidden_states=False,
    #                          training=False)
    #     loss = loss_fn(tar_real, outputs['output'])
    #
    #     if not testing:
    #         val_loss.update_state(loss)
    #     else:
    #         test_loss.update_state(loss)
    #
    #     return outputs
    #
    # summary_res = 1
    # for epoch in range(1000):
    #     train_loss.reset_states()
    #     val_loss.reset_states()
    #
    #     train_step(inp=x[:,:,:1], tar=x[:,:,1:], global_attention_mask=global_attention_mask)
    #     val_step(inp=x_test[:,:,:1], tar=x_test[:,:,1:], global_attention_mask=global_attention_mask[:2,:])
    #
    #     print('TrainLoss: ', train_loss.result().numpy())
    #     print('ValLoss: ', val_loss.result().numpy())

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    b_size = 3
    data_dir = '../../../Data/bach_multiinstrumental/bach_multiinstrumental/Training/Audio'
    target_bin = [1, 2, 4]
    input_bin = [6, 5, 3]
    x = []
    y = []
    nperseg = []
    sample_freq = []
    len_original_freq = []
    # TODO: Need to load data properly, i.e. not just three samples.
    for inp, tar in zip(input_bin, target_bin):
        #   target          input
        # 	1 (0001) from   6 (0110)
        # 	2 (0010) from   5 (0101)
        # 	4 (0100) from   3 (0011)
        sample_freq_i, x_i = wavfile.read('/'.join([data_dir, '01AusmeinesHerz.mid_0_' + str(inp) + '.wav']))
        _, y_i = wavfile.read('/'.join([data_dir, '01AusmeinesHerz.mid_0_' + str(tar) + '.wav']))

        nperseg_i = len(x_i) // ((len(x_i) // sample_freq_i) * 10)  # i.e. each window is (1/10th of a second)

        f, t, Zxx = x_stft = signal.stft(x_i, fs=sample_freq_i, nperseg=nperseg_i)  # 4096
        f_y, t_y, Zyy = y_stft = signal.stft(y_i, fs=sample_freq_i, nperseg=nperseg_i)

        assert all(f == f_y) and all(t == t_y)

        relevant_freq = np.where(f > 12500)[0][0]
        len_original_freq.append(len(f))            # Required for inverse STFT later since we're removing high freqs.
        f = f[:relevant_freq]
        Zxx = Zxx[:relevant_freq, :]
        Zxx = np.abs(Zxx).T
        Zyy = Zyy[:relevant_freq, :]
        Zyy = np.abs(Zyy).T

        x.append(Zxx)
        y.append(Zyy)
        nperseg.append(nperseg_i)
        sample_freq.append(sample_freq_i)

    x = np.array(x)
    y = np.array(y)

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1) and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    scaler = my_scaler_stft()
    scaler.fit(np.concatenate([x, y], axis=0))
    x = scaler.transform(x)
    y = scaler.transform(y)

    # -----------------------------------------------------------------------------------------------------------------
    # Set-up model, optimiser, lr_sched and losses:
    # -----------------------------------------------------------------------------------------------------------------
    max_length = x.shape[1]
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    output_dim = x.shape[-1]
    max_position_embedding = x.shape[1]
    hidden_size = 128
    configuration = LongformerConfig(attention_window=[64, 64, 64],
                                     pad_token_id=1,  # Pad token id (don't change)
                                     hidden_size=hidden_size,
                                     num_hidden_layers=3,
                                     num_attention_heads=4,
                                     intermediate_size=512,  # For intermediate dense layer
                                     hidden_act="tanh",
                                     hidden_dropout_prob=0.,
                                     attention_probs_dropout_prob=0.,
                                     max_position_embeddings=max_position_embedding,  # For positional encoding
                                     initializer_range=0.02,
                                     layer_norm_eps=1e-12)

    longformer = My_TFLongformerModel(configuration, output_dim=output_dim, act_fn_output='sigmoid')
    np.random.seed(4)
    global_attention_mask = np.random.choice([0, 1], (1, tf.shape(x)[1]), p=[.85, .15])
    # global_attention_mask[0, -1] = 1
    global_attention_mask = tf.repeat(global_attention_mask, b_size, axis=0)

    learning_rate = CustomSchedule(d_model=d_model, warmup_steps=4000)
    opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_fn = tf.keras.losses.MeanAbsoluteError()
    # loss_fn = tf.keras.metrics.MeanAbsoluteError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # -----------------------------------------------------------------------------------------------------------------
    # Define the training functionality
    # -----------------------------------------------------------------------------------------------------------------

    # @tf.function
    def train_step(inp, tar, global_attention_mask=None):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]

        with tf.GradientTape() as tape:
            predictions = longformer(inputs_embeds=inp,
                                     inputs_embeds_dec=tar_inp,
                                     global_attention_mask=global_attention_mask,
                                     global_attention_mask_dec=global_attention_mask,
                                     output_attentions=False,
                                     output_hidden_states=False,
                                     training=True)
            predictions = predictions['output']
            loss = loss_fn(tar_real, predictions)
        print(loss)
        print(tf.shape(loss))
        gradients = tape.gradient(loss, longformer.trainable_variables)
        opt.apply_gradients(zip(gradients, longformer.trainable_variables))
        train_loss.update_state(loss)


    @tf.function
    def val_step(inp, tar, testing=False, global_attention_mask=None):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]

        outputs = longformer(inputs_embeds=inp,
                             inputs_embeds_dec=tar_inp,
                             global_attention_mask=global_attention_mask,
                             global_attention_mask_dec=global_attention_mask,
                             output_attentions=False,
                             output_hidden_states=False,
                             training=False)
        loss = loss_fn(tar_real, outputs['output'])

        if not testing:
            val_loss.update_state(loss)
        else:
            test_loss.update_state(loss)

        return outputs


    # -----------------------------------------------------------------------------------------------------------------
    # Train the model
    # -----------------------------------------------------------------------------------------------------------------

    summary_res = 1
    epochs = 1000
    for epoch in range(epochs):
        train_loss.reset_states()
        val_loss.reset_states()

        # Get batches
        x_batches, y_batches = get_batches(x, y, b_size=b_size, shuffle=True, seed=epoch)

        # Set-up training progress bar
        n_batch = len(x_batches)
        print("\nepoch {}/{}".format(epoch + 1, epochs))
        pb_i = Progbar(n_batch * b_size, stateful_metrics=['Loss: '])

        for (batch_num, (x_batch, y_batch)) in enumerate(zip(x_batches, y_batches)):
            x_batch = tf.constant(x_batch, dtype='float32')
            y_batch = tf.constant(y_batch, dtype='float32')

            train_step(inp=x_batch, tar=y_batch, global_attention_mask=global_attention_mask[:tf.shape(x_batch)[0],:])

            # Print progbar
            if batch_num % summary_res == 0:
                values = [('Loss: ', train_loss.result())]
                pb_i.add((batch_num+1)*b_size, values=values)

        print('TrainLoss: ', train_loss.result().numpy())
        print('ValLoss: ', val_loss.result().numpy())

    predictions = longformer(inputs_embeds=x_batch[:1,:,:],
                             inputs_embeds_dec=y_batch[:1,:-1,:],
                             global_attention_mask=global_attention_mask[:1,:],
                             global_attention_mask_dec=global_attention_mask[:1,:],
                             output_attentions=False,
                             output_hidden_states=False,
                             training=False)
    predictions = predictions['output']
    print(loss_fn(y_batch[:, 1:], predictions).numpy())
    predictions = predictions.numpy()
    predictions = scaler.inverse_transform(predictions)
    predictions = tf.squeeze(predictions).numpy()       # Remove dimensions of size == 1.

    # To visualise the results in the time-freq spectrum:
    plt.subplots()
    plt.pcolormesh(t[1:], f[:relevant_freq], predictions.T, vmin=0, vmax=np.max(predictions.T), shading='gouraud')
    plt.title('STFT Magnitude Pred')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    plt.subplots()
    plt.pcolormesh(t, f[:relevant_freq], scaler.inverse_transform(x)[0].T, vmin=0,
                   vmax=np.max(scaler.inverse_transform(x)[0].T), shading='gouraud')
    plt.title('STFT Magnitude Input')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # To perform inverse STFT we need to have the same frequency bins as were produced by the original STFT (i.e.
    # before removing higher frequencies), we therefore pad these values with zeroes:
    predictions = np.concatenate(
        [predictions, np.zeros((predictions.shape[0], len_original_freq[0] - predictions.shape[-1]))], axis=-1)
    _, predictions = signal.istft(predictions.T, fs=sample_freq[-1], nperseg=nperseg[-1])     # Inverse STFT
    predictions = predictions.astype('int16')   # Convert the data to int16 values.
    wavfile.write(r'C:\Users\larsbent\Downloads\predictions.wav', sample_freq[-1], predictions)


    # 'last_hidden_state', 'hidden_states', 'attentions', 'global_attentions'
    #       Note that attention shapes are (bs, n_heads, seq_len, attn_windows + global_attn + 1)

    # attn_probs: shape = (bs, seq_len, num_heads, global_attn + attn_windows + 1)
    # For three global indices and a window length of 4:
    # Note that if the the particular window indices is a global indices --> then 0.
    # If winodow is outside of data range (i.e. wind_-2 and wind_-1 for the first entry) --> 0
    # [glob_1, glob_2, glob_3, window_-2, window_-1, self, wind_+1, wind_+2]
    #
    # global_attn_probs: shape = (bs, num_heads, num_glob_indices, seq_len)
    #   The first global attentions over the entire input is global_attn_probs[0,0,0,:].
    #   0-values indicate the padding (i.e. the last added padding to ensure the seq is a multiple of the wind length)

    # print(output)
    # print(tf.shape(output))



