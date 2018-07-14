#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import torch

from sacred import Ingredient

net = Ingredient('network')


@net.config
def cfg():
    input = [
        {'name': 'reshape', 'shape': [1, 64, 64]},
        {'name': 'conv', 'size_in' : 1, 'size': 16, 'act': 'elu', 'stride': (2, 2), 'kernel': (4, 4), 'ln': True},
        {'name': 'conv', 'size_in' : 16, 'size': 32, 'act': 'elu', 'stride': (2, 2), 'kernel': (4, 4), 'ln': True},
        {'name': 'conv', 'size_in' : 32, 'size': 64, 'act': 'elu', 'stride': (2, 2), 'kernel': (4, 4), 'ln': True},
        {'name': 'reshape', 'shape': [-1]},
        {'name': 'fc', 'size_in' : 4096, 'size': 512, 'act': 'elu', 'ln': True},
    ]
    recurrent = [
        {'name': 'fc', 'size_in' : 512+250+250, 'size': 250, 'act': 'sigmoid', 'ln': True,
         'encoder': [
             {'name': 'fc', 'size_in' : 250, 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'core': [
             {'name': 'fc', 'size_in' : 500, 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'context': [
             {'name': 'fc', 'size_in' : 250, 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'attention': [
             {'name': 'fc', 'size_in' : 250, 'size': 100, 'act': 'tanh', 'ln': True},
             {'name': 'fc', 'size_in' : 100, 'size': 1, 'act': 'sigmoid'},
         ]}
    ]
    output = [
        {'name': 'fc', 'size_in' : 250, 'size': 512, 'act': 'relu', 'ln': True},
        {'name': 'fc', 'size_in' : 512, 'size': 8*8*64, 'act': 'relu', 'ln': True},
        {'name': 'reshape', 'shape': [64, 8, 8]},
        {'name': 'r_conv', 'in_shape' : (8,8), 'size_in' : 64, 'size': 32, 'act': 'relu', 'stride': (2, 2), 'kernel': (5, 5), 'ln': True},
        {'name': 'r_conv', 'in_shape' : (16,16), 'size_in' : 32, 'size': 16, 'act': 'relu', 'stride': (2, 2), 'kernel': (5, 5), 'ln': True},
        {'name': 'r_conv', 'in_shape' : (32,32), 'size_in' : 16, 'size': 1, 'act': 'sigmoid', 'stride': (2, 2), 'kernel': (5, 5)},
        {'name': 'reshape', 'shape': [-1]},
    ]

# encoder decoder pairs


net.add_named_config('enc_dec_84_atari', {
    'input': [
        {'name': 'reshape', 'shape': (84, 84, 1)},
        {'name': 'conv', 'size': 16, 'act': 'elu', 'stride': [4, 4], 'kernel': (8, 8), 'ln': True},
        {'name': 'conv', 'size': 32, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'reshape', 'shape': -1},
        {'name': 'fc', 'size': 250, 'act': 'elu', 'ln': True},
    ],
    'output': [
        {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
        {'name': 'fc', 'size': 10*10*32, 'act': 'relu', 'ln': True},
        {'name': 'reshape', 'shape': (10, 10, 32)},
        {'name': 'r_conv', 'size': 16, 'act': 'relu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True, 'offset': 1},
        {'name': 'r_conv', 'size': 1, 'act': 'sigmoid', 'stride': [4, 4], 'kernel': (8, 8)},
        {'name': 'reshape', 'shape': -1},
    ]})


# recurrent configurations

net.add_named_config('rnn_250', {'recurrent': [{'name': 'rnn', 'size': 250, 'act': 'sigmoid', 'ln': True}]})
net.add_named_config('lstm_250', {'recurrent': [{'name': 'lstm', 'size': 250, 'act': 'sigmoid', 'ln': True}]})

net.add_named_config('r_nem', {
    'recurrent': [
        {'name': 'fc', 'size_in' : 512+250+250, 'size': 250, 'act': 'sigmoid', 'ln': True,
         'encoder': [
             {'name': 'fc', 'size_in' : 250, 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'core': [
             {'name': 'fc', 'size_in' : 500, 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'context': [
             {'name': 'fc', 'size_in' : 250, 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'attention': [
             {'name': 'fc', 'size_in' : 250, 'size': 100, 'act': 'tanh', 'ln': True},
             {'name': 'fc', 'size_in' : 100, 'size': 1, 'act': 'sigmoid'},
         ]}
    ]})


net.add_named_config('r_nem_no_attention', {
    'recurrent': [
        {'name': 'r_nem', 'size': 250, 'act': 'sigmoid', 'ln': True,
         'encoder': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'core': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'context': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'attention': []}
    ]})


net.add_named_config('r_nem_actions', {
    'recurrent': [
        {'name': 'r_nem', 'size': 250, 'act': 'sigmoid', 'ln': True,
         'encoder': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'core': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'context': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'attention': [
             {'name': 'fc', 'size': 100, 'act': 'tanh', 'ln': True},
             {'name': 'fc', 'size': 1, 'act': 'sigmoid'},
         ],
         'actions': [
             {'name': 'fc', 'size': 10, 'act': 'relu', 'ln': True},
         ]}
    ]})


# GENERIC WRAPPERS
class LayerWrapper(torch.nn.Module):
    def __init__(self, spec, name="Wrapper"):
        super(LayerWrapper, self).__init__()
        self._spec = spec
        self._name = name
        if self._spec['name'] == 'fc':
            self._layer = torch.nn.Linear(self._spec['size_in'], self._spec['size'])
        elif self._spec['name'] == 'conv':
            self._layer = torch.nn.Conv2d(self._spec['size_in'], self._spec['size'], self._spec['kernel'], stride=self._spec['stride'], padding=(1,1))
        elif self._spec['name'] == 'r_conv':
            self._layer = torch.nn.Conv2d(self._spec['size_in'], self._spec['size'], self._spec['kernel'], padding=(2,2))

        self._ln = None
        if self._spec.get('ln', None) == True:
            if self._spec['name'] == 'fc':
                self._ln = torch.nn.BatchNorm1d(self._spec['size'])
            elif self._spec['name'] == 'conv':
                self._ln = torch.nn.BatchNorm2d(self._spec['size'])

        self._act = None
        if self._spec.get('act',None) == 'elu':
            self._act = torch.nn.ELU()
        elif self._spec.get('act',None) == 'relu':
            self._act = torch.nn.ReLU()
        elif self._spec.get('act',None) == 'sigmoid':
            self._act = torch.nn.Sigmoid()
        elif self._spec.get('act',None) == 'tanh':
            self._act = torch.nn.Tanh()

        self._transform = None
        if self._spec['name']=='r_conv':
            self._transform = torch.nn.Upsample(scale_factor=self._spec['stride'][0], mode="bilinear")


    def forward(self, input):
        if self._spec['name'] == 'reshape':
            output = input.view([list(input.size())[0]]+self._spec['shape'])
            return output

        if self._spec['name'] == 'r_conv':
            input = self._transform(input)
        output = self._layer(input)
        if self._ln != None:
            output = self._ln(output)
        if self._act!=None:
            output = self._act(output)
        print(output.size())
        return output



# R-NEM CELL
class R_NEM(torch.nn.Module):
    @net.capture
    def __init__(self, K, input, output, recurrent, actions=None, name='NPE'):
        super(R_NEM, self).__init__()
        self._encoder = recurrent[0]["encoder"]
        self._core = recurrent[0]["core"]
        self._context = recurrent[0]["context"]
        self._attention = recurrent[0]["attention"]
        self._actions = actions
        self._recurrent = recurrent
        self._K = K
        self._name = name

        mods = [LayerWrapper(mod) for mod in input]
        self._input_wrapper = torch.nn.Sequential(*mods)

        mods = [LayerWrapper(mod) for mod in output]
        self._output_wrapper = torch.nn.Sequential(*mods)

        mods = [LayerWrapper(mod) for mod in self._encoder]
        self._encoder_wrapper = torch.nn.Sequential(*mods)

        mods = [LayerWrapper(mod) for mod in self._core]
        self._core_wrapper = torch.nn.Sequential(*mods)

        mods = [LayerWrapper(mod) for mod in self._context]
        self._context_wrapper = torch.nn.Sequential(*mods)

        mods = [LayerWrapper(mod) for mod in self._attention]
        self._att_wrapper = torch.nn.Sequential(*mods)

        mods = [LayerWrapper(mod) for mod in self._recurrent]
        self._recurrent_wrapper = torch.nn.Sequential(*mods)

    @property
    def state_size(self):
        return self._recurrent[0]['size']

    @property
    def output_size(self):
        return self._recurrent[0]['size']

    def init_hidden(self, batch_size):
        # variable of size [num_layers*num_directions, b_sz, hidden_sz]
        return torch.autograd.Variable(torch.zeros(batch_size, self.state_size)).cuda() 

    def forward(self, inputs, state):
        b = int(inputs.size()[0]/self._K)
        k = self._K
        m = inputs.size()[1]

        inputs = self._input_wrapper(inputs)

        state1 = self._encoder_wrapper(state)
        state1r = state1.view(b,self._K,-1)
        state1rr = state1.view(b,self._K,1,-1)

        fs = state1rr.repeat(1,1,self._K-1,1)
        state1rl = torch.unbind(state1r,1)

        if k > 1:
            csu = []
            for i in range(k):
                selector = [j for j in range(k) if j != i]
                c = list(np.take(state1rl, selector))  # list of length k-1 of (b, h1)
                c = torch.stack(c, dim=1)     # (b, k-1, h1)
                csu.append(c)

            cs = torch.stack(csu, dim=1)    # (b, k, k-1, h1)   
        else:
            cs = torch.zeros(b, k, k-1, h1)

        cs = cs.view(b*self._K*(self._K-1),-1)
        fs = fs.view(b*self._K*(self._K-1),-1)

        core_out = torch.cat((cs,fs),dim=1)

        core_out = self._core_wrapper(core_out)

        context = self._context_wrapper(core_out)
        contextr = context.view(b*self._K, self._K-1, -1)

        attention = self._att_wrapper(core_out)
        attentionr = attention.view(b*k, k-1, 1)
        effectrsum = torch.sum(torch.mul(attentionr, contextr), dim=1)

        total = torch.cat((state1, effectrsum, inputs), dim=1)

        new_state = self._recurrent_wrapper(total)

        return self._output_wrapper(new_state), new_state

