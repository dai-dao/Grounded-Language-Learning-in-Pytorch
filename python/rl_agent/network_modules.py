import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from collections import namedtuple

class Vision_M(nn.Module):
    def __init__(self):
        '''
            Use the same hyperparameter settings denoted in the paper
        '''
        
        super(Vision_M, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4 )
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2 )
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        
    def forward(self, x):
        # x is input image with shape [3, 84, 84]
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        return out
    

class Language_M(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=128, hidden_size=128):
        '''
            Use the same hyperparameter settings denoted in the paper
        '''
        
        super(Language_M, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim)  
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1, batch_first=True)
        
    def forward(self, x):
        embedded_input = self.embeddings(x)
        out, hn = self.lstm(embedded_input)
        h, c = hn
        
        return h

    def LP(self, x):
        return F.linear(x, self.embeddings.weight)
    

class Mixing_M(nn.Module):
    def __init__(self):
        super(Mixing_M, self).__init__()

    
    def forward(self, visual_encoded, instruction_encoded):
        '''
            Argument:
                visual_encoded: output of vision module, shape [batch_size, 64, 7, 7]
                instruction_encoded: hidden state of language module, shape [batch_size, 1, 128]
        '''
        batch_size = visual_encoded.size()[0]
        visual_flatten = visual_encoded.view(batch_size, -1)
        instruction_flatten = instruction_encoded.view(batch_size, -1)
                
        mixed = torch.cat([visual_flatten, instruction_flatten], dim=1)
        
        return mixed
    
    
class Action_M(nn.Module):
    def __init__(self, batch_size=1, hidden_size=256):
        super(Action_M, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        
        self.lstm_1 = nn.LSTMCell(input_size=3264, hidden_size=256)
        self.lstm_2 = nn.LSTMCell(input_size=256, hidden_size=256)
        
        self.hidden_1 = (Variable(torch.randn(batch_size, hidden_size)).cuda(), 
                        Variable(torch.randn(batch_size, hidden_size)).cuda()) 
        
        self.hidden_2 = (Variable(torch.randn(batch_size, hidden_size)).cuda(), 
                        Variable(torch.randn(batch_size, hidden_size)).cuda()) 
        
    def forward(self, x):
        '''
            Argument:
                x: x is output from the Mixing Module, as shape [batch_size, 1, 3264]
        '''
        # Feed forward
        h1, c1 = self.lstm_1(x, self.hidden_1)
        h2, c2 = self.lstm_2(h1, self.hidden_2)
        
        # Update current hidden state
        self.hidden_1 = (h1, c1)
        self.hidden_2 = (h2, c2)
        
        # Return the hidden state of the upper layer
        return h2
    
    
SavedAction = namedtuple('SavedAction', ['action', 'value'])

class Policy(nn.Module):
    def __init__(self, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        
        self.affine1 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, action_space)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        
        return action_scores, state_values
    
    def tAE(self, action_logits):
        '''
        Temporal Autoencoder sub-task
        Argument:
            action_logits: shape [1, action_space]
        
        Return:
            output has shape: [1, 128]
        
        '''
        bias = torch.unsqueeze(self.action_head.bias, 0).repeat(action_logits.size()[0], 1)
        
        output = action_logits - bias
        output = F.linear(output, torch.transpose(self.action_head.weight, 0, 1))
        
        return output
    
    

class temporal_AutoEncoder(nn.Module):
    def __init__(self, policy_network, vision_module):
        super(temporal_AutoEncoder, self).__init__()
        self.policy_network = policy_network
        self.vision_module = vision_module
    
        self.linear_1 = nn.Linear(128, 64 * 7 * 7)

        self.deconv = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
                        nn.ConvTranspose2d(in_channels=32, out_channels=3,  kernel_size=8, stride=4))
    
    def forward(self, visual_input, logit_action):
        '''
        Argument:
            visual_encoded: output from the visual module, has shape [1, 64, 7, 7]
            logit_action: output action logit from policy, has shape [1, 10]
        '''
        visual_encoded = self.vision_module(visual_input)
        
        action_out = self.policy_network.tAE(logit_action) # [1, 128]
        action_out = self.linear_1(action_out)
        action_out = action_out.view(action_out.size()[0], 64, 7, 7)
        
        combine = torch.mul(action_out, visual_encoded)
        
        out = self.deconv(combine)
        return out        
    
    
class Language_Prediction(nn.Module):
    def __init__(self, language_module):
        super(Language_Prediction, self).__init__()
        self.language_module = language_module
        
        self.vision_transform = nn.Sequential(
                            nn.Linear(64 * 7 * 7, 128),
                            nn.ReLU())
    
    def forward(self, vision_encoded):
            
        vision_encoded_flatten = vision_encoded.view(vision_encoded.size()[0], -1)
        vision_out = self.vision_transform(vision_encoded_flatten)
        
        language_predict = self.language_module.LP(vision_out)
        
        return language_predict
    
    
class RewardPredictor(nn.Module):
    def __init__(self, vision_module, language_module, mixing_module):
        super(RewardPredictor, self).__init__()
        
        self.vision_module = vision_module
        self.language_module = language_module
        self.mixing_module = mixing_module
        self.linear = nn.Linear(3 * (64 * 7 * 7 + 128), 1)
    
    def forward(self, x):
        '''
            x: state including image and instruction, each batch contains 3 image in sequence 
                    with the instruction to be encoded
        '''
        batch_visual = []
        batch_instruction = []
        
        for batch in x:
            visual = [b.visual for b in batch]
            instruction = [b.instruction for b in batch]
            
            batch_visual.append(torch.cat(visual, 0))
            batch_instruction.append(torch.cat(instruction, 0))
        
        batch_visual_encoded = self.vision_module(torch.cat(batch_visual, 0))
        batch_instruction_encoded = self.language_module(torch.cat(batch_instruction, 0))
        
        batch_mixed = self.mixing_module(batch_visual_encoded, batch_instruction_encoded)
        batch_mixed = batch_mixed.view(len(batch_visual), -1)
        
        out = self.linear(batch_mixed)
        return out