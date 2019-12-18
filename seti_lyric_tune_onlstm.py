#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import torch
import torchsnooper
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable


# In[2]:


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


# In[3]:


def embedded_dropout(embed, words, dropout=0.0, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(
            1 - dropout
        ).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = torch.nn.functional.embedding(
        words,
        masked_embed_weight,
        padding_idx,
        embed.max_norm,
        embed.norm_type,
        embed.scale_grad_by_freq,
        embed.sparse,
    )
    return X


# In[4]:


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.0):
        super(LinearDropConnect, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.0:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(self.weight.size(), dtype=torch.uint8)
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.0)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return f.linear(input, self._weight, self.bias)
        else:
            return f.linear(input, self.weight * (1 - self.dropout), self.bias)


# In[5]:


def cumsoftmax(x, dim=-1):
    """
    Cummulative softmax
    """
    return torch.cumsum(f.softmax(x, dim=dim), dim=dim)


# In[26]:


class ONLSTMCell(nn.Module):
    """
    ON-LSTM cell part of the ONLSTMStack.
    Code credits: https://github.com/yikangshen/Ordered-Neurons
    """

    def __init__(self, input_size, hidden_size, n_chunk, dropconnect=0.0):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.n_chunk = n_chunk   # n_chunk 层次数
        self.chunk_size = int(hidden_size / n_chunk)     #chunk_size 广播次数 
        
        
        
        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size, bias=True)
        )
        self.hh = LinearDropConnect(
            hidden_size, hidden_size * 4, bias=True, dropout=dropconnect
        )
        self.drop_weight_modules = [self.hh]

    def forward(self, input, hidden, input_floor, forget_floor, transformed_input=None):
        
        hx, cx = hidden

        if transformed_input is None:
            transformed_input = self.ih(input)
            
        gates = transformed_input + self.hh(hx)
        
        
        outgate, cell, ingate, forgetgate = (
            gates[:,:].view(-1, self.n_chunk * 4, self.chunk_size).chunk(4, 1)
        )
        
        happy_setiment_input, sad_setiment_input = input_floor.chunk(2,1)
        happy_setiment_forget, sad_setiment_forget = forget_floor.chunk(2,1)
        
        
        happy_cingate = cumsoftmax(happy_setiment_input)
        happy_setiment_forget = happy_setiment_forget.flip([1])
        happy_setiment_forget = cumsoftmax(happy_setiment_forget)
        happy_cforgetgate = happy_setiment_forget.flip([1])
        
        sad_setiment_input = sad_setiment_input.flip([1])
        sad_setiment_input = cumsoftmax(sad_setiment_input)
        sad_cingate = sad_setiment_input.flip([1])
        sad_cforgetgate = cumsoftmax(sad_setiment_forget)
        
        cingate = torch.cat([happy_cingate,sad_cingate],1)
        cforgetgate = torch.cat([happy_cforgetgate,sad_cforgetgate],1)
        
        
        
        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]
        
        ingate = f.sigmoid(ingate)
        forgetgate = f.sigmoid(forgetgate)
        cell = f.tanh(cell)
        outgate = f.sigmoid(outgate)
        
        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell
        
        hy = outgate * f.tanh(cy)
        return hy.view(-1, self.hidden_size), cy, (distance_cforget, distance_cin)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            weight.new(bsz, self.hidden_size).zero_(),
            weight.new(bsz, self.n_chunk, self.chunk_size).zero_(),
        )

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


# In[30]:


class tune_ONLSTMCell(nn.Module):
    """
    ON-LSTM cell part of the ONLSTMStack.
    Code credits: https://github.com/yikangshen/Ordered-Neurons
    """

    def __init__(self, input_size, hidden_size, n_chunk, dropconnect=0.0):
        super(tune_ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.n_chunk = n_chunk   # n_chunk 层次数
        self.chunk_size = int(hidden_size / n_chunk)     #chunk_size 广播次数 
        
        
        
        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size, bias=True)
        )
        self.hh = LinearDropConnect(
            hidden_size, hidden_size * 4, bias=True, dropout=dropconnect
        )
        self.drop_weight_modules = [self.hh]

    def forward(self, input, hidden, input_floor, forget_floor, transformed_input=None):
        
        hx, cx = hidden

        if transformed_input is None:
            transformed_input = self.ih(input)
            
        gates = transformed_input + self.hh(hx)
        
        
        outgate, cell, ingate, forgetgate = (
            gates[:,:].view(-1, self.n_chunk * 4, self.chunk_size).chunk(4, 1)
        )
        
        
        
        cingate = input_floor
        cforgetgate = forget_floor
        
        
        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]
        
        ingate = f.sigmoid(ingate)
        forgetgate = f.sigmoid(forgetgate)
        cell = f.tanh(cell)
        outgate = f.sigmoid(outgate)
        
        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell
        
        hy = outgate * f.tanh(cy)
        return hy.view(-1, self.hidden_size), cy

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            weight.new(bsz, self.hidden_size).zero_(),
            weight.new(bsz, self.n_chunk, self.chunk_size).zero_(),
        )

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


# In[50]:


class ONLSTMStack(nn.Module):
    """
    ON-LSTM encoder composed of multiple ON-LSTM layers. Each layer is constructed
    through ONLSTMCell structures.
    Code credits: https://github.com/yikangshen/Ordered-Neurons
    """

    def __init__(
        self,
        layer_sizes,
        n_chunk,
        dropout=0.0,
        dropconnect=0.0,
        dropouti=0.5,
        dropoutw=0.1,
        dropouth=0.3
    ):
        
        super(ONLSTMStack, self).__init__()
        self.layer_sizes = layer_sizes
        # lyric_onlstm
        self.cells = nn.ModuleList(
            [
                ONLSTMCell(layer_sizes[i], layer_sizes[i + 1], n_chunk, dropconnect=dropconnect)
                for i in range(len(layer_sizes) - 1)
            ]
        )
        
        # tune_onlstm 
        self.tune_cells = nn.ModuleList(
            [
                tune_ONLSTMCell(layer_sizes[i], layer_sizes[i + 1], n_chunk, dropconnect=dropconnect)
                for i in range(len(layer_sizes) - 1)
            ]
        )
        
        self.n_chunk = n_chunk
        self.cells = self.cells.cuda()
        self.lockdrop = LockedDropout()
        
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        
        self.sizes = layer_sizes
        #self.embedder = embedder
        #dim = self.embedder.token_embedder_words.weight.shape
        self.emb = nn.Embedding(12, n_chunk)
        
        self.dropoutw = dropoutw
        
        self.MLP = nn.Sequential(nn.Linear(layer_sizes[-1]*2, 4),
                                 nn.Softmax())
        
        

    def get_input_dim(self):
        return self.layer_sizes[0]

    def get_output_dim(self):
        return self.layer_sizes[-1]

    def init_hidden(self, bsz):
        return [c.init_hidden(bsz) for c in self.cells], [c.init_hidden(bsz) for c in self.tune_cells] 

    def forward(self, input,sentiment_floor, tune_input, task=None):
        batch_size = input.size()[0]
        hidden,tune_hidden = self.init_hidden(batch_size)
#         hidden = hidden.cuda()
#         tune_hidden = tune_hidden.cuda()
        return self.forward_actual(input, hidden, tune_hidden, sentiment_floor, tune_input)


    
    @torchsnooper.snoop()
    def forward_actual(self, input, hidden, tune_hidden, sentiment_floor, tune_input):
        
        # lyric input
        batch_size,length, _ = input.size()
        input = input.transpose(0,1)  # song_length , batch , 768(lyric_cls dims)
        input = self.lockdrop(input, self.dropout)
        sentiment_floor = sentiment_floor.t()
        
        # tune input
        tune_input = tune_input.transpose(0,1)  # song_length , batch, tune_dims
        tune_input = self.lockdrop(tune_input, self.dropout)
        
        # is training ?
        if self.training:
            for c in self.cells:
                c.sample_masks()
                
        if self.training:
            for c in self.tune_cells:
                c.sample_masks()
        
        # initialize
        prev_state = list(hidden)
        prev_layer = input
        
        tune_prev_state = list(tune_hidden)
        tune_prev_layer = tune_input
        
        
        #
        #raw_outputs = []
        #lyric_outputs = []
        
        
        
        for l in range(len(self.cells)):
            # 
            curr_layer = [None] * length
            tune_curr_layer = [None] * length
            
            
            # input
            t_input = self.cells[l].ih(prev_layer.cuda())
            forget_floor = torch.zeros([batch_size, self.n_chunk]).cuda()
            
            tune_input = self.tune_cells[l].ih(tune_prev_layer)
            
            
            # one onlstm-cell processing
            for t in range(length):
                input_floor = embedded_dropout(self.emb, sentiment_floor[t])
                #sentiment_floor = self.lockdrop(input, self.dropouti)
                
                hidden, cell, lyric_gate = self.cells[l](None, prev_state[l], transformed_input=t_input[t],
                                               input_floor = input_floor , forget_floor = forget_floor)
                
                # tune sentiment layer vector only relys on lyric sentiment layer vector 
                tune_hidden, tune_cell  = self.tune_cells[l](None, tune_prev_state[l], transformed_input=tune_input[t],
                                                             input_floor = lyric_gate[0] , forget_floor = lyric_gate[1] )
                
                #  overwritten every timestep
                prev_state[l] = hidden, cell  
                curr_layer[t] = hidden
                
                tune_prev_state[l] = tune_hidden , tune_cell
                tune_curr_layer[t] = tune_hidden
                
                
                #
                forget_floor = input_floor

                
            prev_layer = torch.stack( curr_layer)
            tune_prev_layer = torch.stack(tune_curr_layer)
            
            #raw_outputs.append(prev_layer)
            
            if l < len(self.cells) - 1:
                prev_layer = self.lockdrop(prev_layer, self.dropouth)
                
            if l < len(self.tune_cells) - 1:
                tune_prev_layer = self.lockdrop(tune_prev_layer, self.dropouth)
                
            #outputs.append(prev_layer)
        
        lyric_output = prev_layer[-1,:,:]  #(bz, hidden_size)
        tune_out_put = tune_prev_layer[-1,:,:] # last layer (bz, hz)
        
        output = torch.cat([lyric_output,tune_out_put],1)
        output = self.MLP(output).squeeze(1)
        
        return output


# In[51]:


lyric = torch.Tensor(4, 4, 24)
tune = torch.Tensor(4, 4, 24)
tree_posi = torch.randint(-5,5 , [4, 12])
lstm = ONLSTMStack([24, 48, 24], n_chunk=12)


# In[52]:


output = lstm(lyric, tree_posi, tune)


# In[17]:


aa = torch.rand([1,5])
bb = torch.rand([1,5])
cc = [None] * 2


# In[18]:


cc[0] = aa
cc[1] = bb


# In[20]:


dd = torch.stack(cc)


# In[22]:


dd.shape


# In[ ]:




