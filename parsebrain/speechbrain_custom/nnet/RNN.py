import torch
import logging
import torch.nn as nn
from speechbrain.nnet.attention import (
    ContentBasedAttention,
    LocationAwareAttention,
    KeyValueAttention,
)
from torch import Tensor
from typing import Optional

from speechbrain.nnet.RNN import GRUCell, LSTMCell, RNNCell

# Speechbrain code, just adding the possibility of setting the hidden state to something.


class AttentionalRNNDecoder(nn.Module):
    """This function implements RNN decoder model with attention.
    This function implements different RNN models. It accepts in enc_states
    tensors formatted as (batch, time, fea). In the case of 4d inputs
    like (batch, time, fea, channel) the tensor is flattened in this way:
    (batch, time, fea*channel).
    Arguments
    ---------
    rnn_type : str
        Type of recurrent neural network to use (rnn, lstm, gru).
    attn_type : str
        type of attention to use (location, content).
    hidden_size : int
        Number of the neurons.
    attn_dim : int
        Number of attention module internal and output neurons.
    num_layers : int
        Number of layers to employ in the RNN architecture.
    input_shape : tuple
        Expected shape of an input.
    input_size : int
        Expected size of the relevant input dimension.
    nonlinearity : str
        Type of nonlinearity (tanh, relu). This option is active for
        rnn and ligru models only. For lstm and gru tanh is used.
    re_init : bool
        It True, orthogonal init is used for the recurrent weights.
        Xavier initialization is used for the input connection weights.
    normalization : str
        Type of normalization for the ligru model (batchnorm, layernorm).
        Every string different from batchnorm and layernorm will result
        in no normalization.
    scaling : float
        A scaling factor to sharpen or smoothen the attention distribution.
    channels : int
        Number of channels for location-aware attention.
    kernel_size : int
        Size of the kernel for location-aware attention.
    bias : bool
        If True, the additive bias b is adopted.
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    Example
    -------
    >>> enc_states = torch.rand([4, 10, 20])
    >>> wav_len = torch.rand([4])
    >>> inp_tensor = torch.rand([4, 5, 6])
    >>> net = AttentionalRNNDecoder(
    ...     rnn_type="lstm",
    ...     attn_type="content",
    ...     hidden_size=7,
    ...     attn_dim=5,
    ...     num_layers=1,
    ...     enc_dim=20,
    ...     input_size=6,
    ... )
    >>> out_tensor, attn = net(inp_tensor, enc_states, wav_len)
    >>> out_tensor.shape
    torch.Size([4, 5, 7])
    """

    def __init__(
        self,
        rnn_type,
        attn_type,
        hidden_size,
        attn_dim,
        num_layers,
        enc_dim,
        input_size,
        nonlinearity="relu",
        re_init=True,
        normalization="batchnorm",
        scaling=1.0,
        channels=None,
        kernel_size=None,
        bias=True,
        dropout=0.0,
    ):
        super(AttentionalRNNDecoder, self).__init__()

        self.rnn_type = rnn_type.lower()
        self.attn_type = attn_type.lower()
        self.hidden_size = hidden_size
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.scaling = scaling
        self.bias = bias
        self.dropout = dropout
        self.normalization = normalization
        self.re_init = re_init
        self.nonlinearity = nonlinearity

        # only for location-aware attention
        self.channels = channels
        self.kernel_size = kernel_size

        # Combining the context vector and output of rnn
        self.proj = nn.Linear(self.hidden_size + self.attn_dim, self.hidden_size)

        if self.attn_type == "content":
            self.attn = ContentBasedAttention(
                enc_dim=enc_dim,
                dec_dim=self.hidden_size,
                attn_dim=self.attn_dim,
                output_dim=self.attn_dim,
                scaling=self.scaling,
            )

        elif self.attn_type == "location":
            self.attn = LocationAwareAttention(
                enc_dim=enc_dim,
                dec_dim=self.hidden_size,
                attn_dim=self.attn_dim,
                output_dim=self.attn_dim,
                conv_channels=self.channels,
                kernel_size=self.kernel_size,
                scaling=self.scaling,
            )

        elif self.attn_type == "keyvalue":
            self.attn = KeyValueAttention(
                enc_dim=enc_dim,
                dec_dim=self.hidden_size,
                attn_dim=self.attn_dim,
                output_dim=self.attn_dim,
            )

        else:
            raise ValueError(f"{self.attn_type} is not implemented.")

        self.drop = nn.Dropout(p=self.dropout)

        # set dropout to 0 when only one layer
        dropout = 0 if self.num_layers == 1 else self.dropout

        # using cell implementation to reduce the usage of memory
        if self.rnn_type == "rnn":
            cell_class = RNNCell
        elif self.rnn_type == "gru":
            cell_class = GRUCell
        elif self.rnn_type == "lstm":
            cell_class = LSTMCell
        else:
            raise ValueError(f"{self.rnn_type} not implemented.")

        kwargs = {
            "input_size": input_size + self.attn_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bias": self.bias,
            "dropout": dropout,
            "re_init": self.re_init,
        }
        if self.rnn_type == "rnn":
            kwargs["nonlinearity"] = self.nonlinearity

        self.rnn = cell_class(**kwargs)

    def forward_step(self, inp, hs, c, enc_states, enc_len):
        """One step of forward pass process.
        Arguments
        ---------
        inp : torch.Tensor
            The input of current timestep.
        hs : torch.Tensor or tuple of torch.Tensor
            The cell state for RNN.
        c : torch.Tensor
            The context vector of previous timestep.
        enc_states : torch.Tensor
            The tensor generated by encoder, to be attended.
        enc_len : torch.LongTensor
            The actual length of encoder states.
        Returns
        -------
        dec_out : torch.Tensor
            The output tensor.
        hs : torch.Tensor or tuple of torch.Tensor
            The new cell state for RNN.
        c : torch.Tensor
            The context vector of the current timestep.
        w : torch.Tensor
            The weight of attention.
        """
        cell_inp = torch.cat([inp, c], dim=-1)
        cell_inp = self.drop(cell_inp)
        cell_out, hs = self.rnn(cell_inp, hs)

        c, w = self.attn(enc_states, enc_len, cell_out)
        dec_out = torch.cat([c, cell_out], dim=1)
        dec_out = self.proj(dec_out)

        return dec_out, hs, c, w

    def forward(self, inp_tensor, enc_states, wav_len, hidden_state=None):
        """This method implements the forward pass of the attentional RNN decoder.
        Arguments
        ---------
        inp_tensor : torch.Tensor
            The input tensor for each timesteps of RNN decoder.
        enc_states : torch.Tensor
            The tensor to be attended by the decoder.
        wav_len : torch.Tensor
            This variable stores the relative length of wavform.
        hidden_state: torch.Tensor
            The tensor of the initial hidden state of the RNN. Used to give global idea of the content of the sentence.
        Returns
        -------
        outputs : torch.Tensor
            The output of the RNN decoder.
        attn : torch.Tensor
            The attention weight of each timestep.

        """
        # calculating the actual length of enc_states
        enc_len = torch.round(enc_states.shape[1] * wav_len).long()

        # initialization
        self.attn.reset()
        c = torch.zeros(enc_states.shape[0], self.attn_dim, device=enc_states.device)
        hs = hidden_state

        # store predicted tokens
        outputs_lst, attn_lst = [], []
        for t in range(inp_tensor.shape[1]):
            outputs, hs, c, w = self.forward_step(
                inp_tensor[:, t], hs, c, enc_states, enc_len
            )
            outputs_lst.append(outputs)
            attn_lst.append(w)

        # [B, L_d, hidden_size]
        outputs = torch.stack(outputs_lst, dim=1)

        # [B, L_d, L_e]
        attn = torch.stack(attn_lst, dim=1)

        return outputs, attn
