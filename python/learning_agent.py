from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
from rl_agent.actor_critic import RL_Agent
import deepmind_lab

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--length', type=int, default=10000,
                    help='Number of steps to run the agent')
parser.add_argument('--width', type=int, default=84,
                    help='Horizontal size of the observations')
parser.add_argument('--height', type=int, default=84,
                    help='Vertical size of the observations')
parser.add_argument('--fps', type=int, default=60,
                    help='Number of frames per second')
parser.add_argument('--runfiles_path', type=str, default=None,
                    help='Set the runfiles path to find DeepMind Lab data')
parser.add_argument('--level_script', type=str, default='language_learning',
                    help='The environment level script to load')

parser.add_argument('--tau', type=int, default=0.99, help='Training hyperparameter')
parser.add_argument('--gamma', type=int, default=0.99, help='Discounted factor')
parser.add_argument('--clip-grad-norm', type=int, default=1.0, help='Clip gradient')
parser.add_argument('--num-episodes', type=int, default=100, help='Number of training episodes')

args = parser.parse_args()
if args.runfiles_path:
  deepmind_lab.set_runfiles_path(args.runfiles_path)


# Create environment
env = deepmind_lab.Lab( 
    args.level_script, ['RGB_INTERLACED', 'ORDER'],
    config={
        'fps': str(args.fps),
        'width': str(args.width),
        'height': str(args.height)
    })

# Start the Reinforcement Learning agent
agent = RL_Agent(env, args)
# Train the agent
agent.train()
