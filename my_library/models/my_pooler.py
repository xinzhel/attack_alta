from typing import Union, Optional, Dict, Any

import torch

from allennlp.common import FromParams

from allennlp.modules.transformer.transformer_module import TransformerModule

from allennlp.modules.transformer.activation_layer import ActivationLayer
from allennlp.modules.transformer.attention_module import SelfAttention
from allennlp.modules.transformer.output_layer import OutputLayer
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

from overrides import overrides
import torch
from torch import nn




class MyAttentionLayer(torch.nn.Module, FromParams):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        scoring_func: str = "scaled_dot_product",
    ):
        super().__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_dropout, scoring_func)
        self.output = OutputLayer(hidden_size, hidden_size, hidden_dropout)

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.BoolTensor,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: bool = False,
    ):
        """
        input_tensor : `torch.Tensor`
            Shape `batch_size x seq_len x hidden_dim`
        attention_mask : `torch.BoolTensor`, optional
            Shape `batch_size x seq_len`
        head_mask : `torch.BoolTensor`, optional
        output_attentions : `bool`
            Whether to also return the attention probabilities, default = `False`
        """

        if encoder_hidden_states is not None:
            attention_mask = encoder_attention_mask

        self_output = self.self(
            input_tensor,
            encoder_hidden_states,
            encoder_hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_output[0], input_tensor)
        outputs = (attention_output,) + self_output[1:]  # add attentions if we output them
        return outputs

    @classmethod
    def _get_input_arguments(
        cls,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        submodules = cls._get_mapped_submodules(pretrained_module, source, mapping)

        final_kwargs = {}

        final_kwargs["hidden_size"] = submodules["self.query"].in_features
        final_kwargs["num_attention_heads"] = submodules["self"].num_attention_heads
        final_kwargs["attention_dropout"] = submodules["self.dropout"].p
        final_kwargs["hidden_dropout"] = submodules["output.dropout"].p

        final_kwargs.update(**kwargs)

        return final_kwargs



class TransformerEncoder(torch.nn.Module, FromParams):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        activation: Union[str, torch.nn.Module] = "relu",
        # add_cross_attention: bool = False,
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self._add_cross_attention = add_cross_attention

        self.attention = MyAttentionLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
        )

        # if add_cross_attention:
        #     self.cross_attention = AttentionLayer(
        #         hidden_size=hidden_size,
        #         num_attention_heads=num_attention_heads,
        #         attention_dropout=attention_dropout,
        #         hidden_dropout=hidden_dropout,
        #     )

        self.intermediate = ActivationLayer(
            hidden_size=hidden_size, intermediate_size=intermediate_size, activation=activation
        )
        self.output = OutputLayer(
            input_size=intermediate_size, hidden_size=hidden_size, dropout=hidden_dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        # encoder_hidden_states: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        hidden_states : `torch.Tensor`
            Shape `batch_size x seq_len x hidden_dim`
        attention_mask : `torch.BoolTensor`, optional
            Shape `batch_size x seq_len`
        head_mask : `torch.BoolTensor`, optional
        encoder_hidden_states : `torch.Tensor`, optional
        encoder_attention_mask : `torch.Tensor`, optional
        output_attentions : `bool`
            Whether to also return the attention probabilities, default = `False`
        """
        # self-attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0] # Shape `batch_size x seq_len x hidden_dim`
        # outputs = attention_outputs[1:]  # add self attentions if we output attention weights
        # activation
        # intermediate_output = self.intermediate(attention_output) # Shape `batch_size x seq_len x intermediate_dim`
        # layer norm + residual
        # layer_output = self.output(intermediate_output, attention_output) # Shape `batch_size x seq_len x hidden_dim`
        # outputs = (layer_output,) + outputs # same as `layer_output`
        return attention_output

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# Basically same as bert_pooler,
# However, this module is used to deal with the case of 
# pretrained models not having a pooler, e.g. distil-bert-cased
# so we use config information to construct it
@Seq2VecEncoder.register("my_pooler")
class MyPooler(Seq2VecEncoder):
    def __init__(
        self,
        pretrained_model: str,
        *,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        requires_grad: bool = True,
        dropout: float = 0.0,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        from allennlp.common import cached_transformers

        model = cached_transformers.get(
            pretrained_model,
            False,
            override_weights_file,
            override_weights_strip_prefix,
            **(transformer_kwargs or {}),
        )

        self._dropout = torch.nn.Dropout(p=dropout)

        import copy

        self.pooler = Pooler(model.config)
        for param in self.pooler.parameters():
            param.requires_grad = requires_grad
        self._embedding_dim = model.config.hidden_size

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(
        self, tokens: torch.Tensor, mask: torch.BoolTensor = None, num_wrapping_dims: int = 0
    ):
        pooler = self.pooler
        for _ in range(num_wrapping_dims):
            from allennlp.modules import TimeDistributed

            pooler = TimeDistributed(pooler)
        pooled = pooler(tokens)
        pooled = self._dropout(pooled)
        return pooled

