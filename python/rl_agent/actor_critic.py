import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple
from utils import *
from model import Model


def _action(*entries):
    return np.array(entries, dtype=np.intc)

class RL_Agent(object):
    def __init__(self, env, args):
        self.ACTIONS = [
            _action(-20, 0, 0, 0, 0, 0, 0),
            _action(20, 0, 0, 0, 0, 0, 0),
            _action(0, 10, 0, 0, 0, 0, 0),
            _action(0, -10, 0, 0, 0, 0, 0),
            _action(0, 0, -1, 0, 0, 0, 0),
            _action(0, 0, 1, 0, 0, 0, 0),
            _action(0, 0, 0, 1, 0, 0, 0),
            _action(0, 0, 0, -1, 0, 0, 0),
            _action(0, 0, 0, 0, 1, 0, 0),
            _action(0, 0, 0, 0, 0, 1, 0),
            _action(0, 0, 0, 0, 0, 0, 1)]
        
        self.model = Model(len(self.ACTIONS))
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.0001)
        self.model.cuda()
        
        self.memory = ReplayMemory(200)
        self.args = args
        self.env = env
        
       
    def optimize_model(self, values, log_probs, rewards, entropies):
        R = values[-1]
        gae = torch.zeros(1, 1).type(torch.cuda.FloatTensor)

        # Base A3C Loss
        policy_loss, value_loss = 0, 0

        # Performing update
        for i in reversed(range(len(rewards))):
            # Value function loss
            R = self.args.gamma * R + rewards[i]
            value_loss = value_loss + 0.5 * (R - values[i]).pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + self.args.gamma * \
                    values[i + 1].data - values[i].data
            gae = gae * self.args.gamma * self.args.tau + delta_t

            # Computing policy loss
            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        # Auxiliary loss
        language_prediction_loss = 0 
        tae_loss = 0
        reward_prediction_loss = 0
        value_replay_loss = 0

        # Non-skewed sampling from experience buffer
        auxiliary_sample = self.memory.sample(11)
        auxiliary_batch = Transition(*zip(*auxiliary_sample))

        # Language Prediction Loss
        # TODO #

        # TAE Loss
        visual_input = auxiliary_batch.state[:10]
        visual_input = torch.cat([t.visual for t in visual_input], 0)

        visual_target = auxiliary_batch.state[1:11]
        visual_target = torch.cat([t.visual for t in visual_target], 0)

        action_logit = torch.cat(auxiliary_batch.action_logit[:10], 0)

        tae_output = self.model.tAE(visual_input, action_logit)
        tae_loss = torch.sum((tae_output - visual_target).pow(2))

        # Skewed-Sampling from experience buffer # TODO
        skewed_sample = self.memory.sample(31)  # memory.skewed_sample(31)
        skewed_batch = Transition(*zip(*skewed_sample))

        # Reward Prediction loss
        batch_rp_input = []
        batch_rp_output = []

        for i in range(10):
            rp_input = skewed_batch.state[i : i+3]
            rp_output = skewed_batch.reward[i+3]

            batch_rp_input.append(rp_input)
            batch_rp_output.append(rp_output)

        rp_predicted = self.model.reward_predictor(batch_rp_input)

        self.optimizer.zero_grad()
        reward_prediction_loss = \
                torch.sum((rp_predicted - Variable(torch.cuda.FloatTensor(batch_rp_output))).pow(2))

        # Value function replay
        index = np.random.randint(0, 10)
        R_vr = auxiliary_batch.value[index+1] * self.args.gamma + auxiliary_batch.reward[index]
        value_replay_loss = 0.5 * torch.squeeze((R_vr - auxiliary_batch.value[index]).pow(2))

        # Back-propagation
        total_loss = (policy_loss + 0.5 * value_loss +  \
                     reward_prediction_loss +  tae_loss +  \
                     language_prediction_loss + value_replay_loss).cuda()
                
        total_loss.backward(retain_variables=True)
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip_grad_norm)

        # Apply updates
        self.optimizer.step()
        return total_loss.cpu().data.numpy()
        

    def process_state(self, state):
        img = np.expand_dims(np.transpose(state['RGB_INTERLACED'], (2, 0, 1)), 0)
        order = np.expand_dims((state['ORDER']), 0)
        
        img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        order = torch.from_numpy(order).type(torch.cuda.LongTensor)
        
        return State(Variable(img), Variable(order))

    def train(self):

        for episode in range(10):
            print("STARTED EPISODE", episode)
            self.env.reset()
            
            state = self.process_state(self.env.observations())
            total_loss = 0
            episode_length = 0
            while True:
                episode_length += 1
                
                values = []
                log_probs = []
                rewards = []
                entropies = []
                
                logit, value = self.model(state)
                
                # Calculate entropy from action probability distribution
                prob = F.softmax(logit)
                log_prob = F.log_softmax(logit)
                entropy = -(log_prob * prob).sum(1)
                entropies.append(entropy)

                # Take an action from distribution
                action = prob.multinomial().data
                log_prob = log_prob.gather(1, Variable(action))       

                # Perform the action on the environment
                reward = self.env.step(self.ACTIONS[action.cpu().numpy()[0][0]], num_steps=4)
                if not self.env.is_running():
                    self.env.reset() # Environment timed-out 
                next_state = self.process_state(self.env.observations())
                
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                # Push to experience replay buffer
                # THERE IS NO Terminal state in the buffer, ONLY transition
                self.memory.push(state, logit, next_state, reward, value)

                if self.memory.full():
                    _, final_value = self.model(next_state)
                    values.append(final_value)
                    next_state = self.process_state(self.env.observations())
                    
                    # Perform optimization when memory is full
                    loss = self.optimize_model(values, log_probs, rewards, entropies)
                    total_loss += loss
                    
                    # Clear memory 
                    self.memory.clear()
                    
                # move to next state
                state = next_state
                
                # Go to next episode
                if episode_length >= self.args.length / 10:
                    total_loss /= self.args.length / 10
                    print('Episode {} / {} has completed. Episode loss is {}'.
                                       format(episode, self.args.num_episodes, total_loss))
                    break
