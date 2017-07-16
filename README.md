## Implementation of the paper Grounded Language Learning in a Simulated 3D World in PyTorch

To run the custom learning environment, run the following command:

`` bazel run :game level_script language_learning ``

### Milestones:

**1.** Setting up agent environment:
- [x] Generate custom map
- [ ] Generate custom scoring for objects
- [ ] Generate natural language commands

**2.** Build the RL model in pytorch
- [x] Baseline Actor-Critic RL agent
- [x] Neural Network Modules Implementation (Vison, Language, Mixing, Action, Policy)
- [x] Tested baseline implementation
- [ ] Implement auxiliary tasks
- [ ] Train agent on custom environment


