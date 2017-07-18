import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple
from utils import *
from model import Model


# Training parameter
env = FakeEnvironment()
state = env.reset()

model = Model()
memory = ReplayMemory(200)

optimizer = optim.RMSprop(model.parameters())

# Hyperparameters
gamma = 0.99
tau = 0.99

values = []
log_probs = []
rewards = []
entropies = []
episode_length = 0
num_episodes = 10


def optimize_model(values, log_probs, rewards):
    # Wait until memory buffer is big enough
    if len(memory) < 50:
        return 
    
    optimizer.zero_grad()
    
    R = values[-1]
    gae = torch.zeros(1, 1)
    
    # Base A3C Loss
    policy_loss, value_loss = 0, 0

    # Performing update
    for i in reversed(range(len(rewards))):
        # Value function loss
        R = gamma * R + rewards[i]
        value_loss = value_loss + 0.5 * (R - values[i]).pow(2)

        # Generalized Advantage Estimataion
        delta_t = rewards[i] + gamma * \
                values[i + 1].data - values[i].data
        gae = gae * gamma * tau + delta_t

        # Computing policy loss
        policy_loss = policy_loss - \
            log_probs[i] * Variable(gae) - 0.01 * entropies[i]

    # Auxiliary loss
    language_prediction_loss = 0 
    tae_loss = 0
    reward_prediction_loss = 0
    value_replay_loss = 0

    # Non-skewed sampling from experience buffer
    auxiliary_sample = memory.sample(11)
    auxiliary_batch = Transition(*zip(*auxiliary_sample))

    # Language Prediction Loss
    # TODO #
    
    # TAE Loss
    visual_input = auxiliary_batch.state[:10]
    visual_input = torch.cat([t.visual for t in visual_input], 0)

    visual_target = auxiliary_batch.state[1:11]
    visual_target = torch.cat([t.visual for t in visual_target], 0)
    
    action_logit = torch.cat(auxiliary_batch.action_logit[:10], 0)
        
    tae_output = model.tAE(visual_input, action_logit)
    tae_loss = torch.sum((tae_output - visual_target).pow(2))
    
    # Skewed-Sampling from experience buffer # TODO
    skewed_sample = memory.sample(31)  # memory.skewed_sample(31)
    skewed_batch = Transition(*zip(*skewed_sample))
    
    # Reward Prediction loss
    batch_rp_input = []
    batch_rp_output = []
    
    for i in range(10):
        rp_input = skewed_batch.state[i : i+3]
        rp_output = skewed_batch.reward[i+3]
            
        batch_rp_input.append(rp_input)
        batch_rp_output.append(rp_output)
            
    rp_predicted = model.reward_predictor(batch_rp_input)
    
    reward_prediction_loss = \
            torch.sum((rp_predicted - Variable(torch.FloatTensor(batch_rp_output))).pow(2))
    
    # Value function replay
    index = np.random.randint(0, 10)
    R_vr = auxiliary_batch.value[index+1] * gamma + auxiliary_batch.reward[index]
    value_replay_loss = 0.5 * torch.squeeze((R_vr - auxiliary_batch.value[index]).pow(2))
            
    # Back-propagation
    (policy_loss + 0.5 * value_loss + reward_prediction_loss + 
                         tae_loss + language_prediction_loss + 
                         value_replay_loss).backward(retain_variables=True)
    torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    
    # Apply updates
    optimizer.step()


def train():
    for episode in range(num_episodes):
        while True:
            episode_length += 1

            logit, value = model(state)

            # Calculate entropy from action probability distribution
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            # Take an action from distribution
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            # Perform the action on the environment
            next_state, reward, done, _ = env.step(action.numpy())

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            # Push to experience replay buffer
            # THERE IS NO Terminal state in the buffer, ONLY transition
            # THere'll be NO resetting the MEMORY Buffer
            memory.push(state, logit, next_state, reward, value)

            if done:
                final_value = Variable(torch.zeros(1, 1))
            elif episode_length >= 200:
                _, final_value = model(next_state)

            done = done or episode_length >= 200
            if done:
                values.append(final_value)
                episode_length = 0
                next_state = env.reset()

            # move to next state
            state = next_state

            # Optimize model
            optimize_model(values, log_probs, rewards)

            if done:
                break