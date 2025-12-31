import numpy as np
import os
from PPO_model import *

class MultiAgentPPO:
    def __init__(self, action_space_config):
        """
        :param action_space_config:
        example: {
            'gamma_func': {'type': 'continuous', 'min': 0.01, 'max': 0.5},
            'n_ei_candidates': {'type': 'discrete', 'options': [10, 20, 30, 50]}
        }
        """
        self.agents = {}
        self.action_space = action_space_config

        # 为每个动作创建独立的PPO agent
        for param_name, config in action_space_config.items():
            state_dim = 6
            if config['type'] == 'continuous':
                action_dim = 1
            else:
                action_dim = len(config['options'])

            # Trying to load existing model.
            model_path = f"PPO_model_save_dir\\ppo_{param_name}.pth"
            if os.path.exists(model_path):
                print('Already exist model of {}'.format(param_name))
                agent = PPO(state_dim, action_dim, lr_actor=0.001, lr_critic=0.003,
                            gamma=0.99, K_epochs=4, eps_clip=0.2,
                            has_continuous_action_space=(config['type'] == 'continuous'),action_std_init=0.6)
                agent.load(model_path)
            else:
                print('Create model of {}'.format(param_name))
                agent = PPO(state_dim, action_dim, lr_actor=0.001, lr_critic=0.003,
                            gamma=0.99, K_epochs=4, eps_clip=0.2,
                            has_continuous_action_space=(config['type'] == 'continuous'),action_std_init=0.6)

            self.agents[param_name] = agent

    def get_actions(self, state):
        """Acquiring actions."""
        actions = {}
        if isinstance(state, dict):
            state = list(state.values())

        state_tensor = torch.FloatTensor(np.array(state, dtype=np.float32)).to(device)

        for param_name, agent in self.agents.items():
            config = self.action_space[param_name]

            if config['type'] == 'continuous':
                # Continuous space.
                action = agent.select_action(state_tensor)
                action = np.clip(action, -1.0, 1.0)
                scaled_action = config['min'] + (config['max'] - config['min']) * (action + 1) / 2
                actions[param_name] = float(scaled_action)
            else:
                # Discrete space.
                action_idx = agent.select_action(state_tensor)
                actions[param_name] = config['options'][action_idx]

        return actions

    def update_agents(self, round, rewards,is_terminal):
        """Update all agent."""
        rewards_tensor = torch.FloatTensor(rewards).to(device)

        # Update buffer.
        for param_name, agent in self.agents.items():
            if agent.action_std > 0.2:
                agent.set_action_std(0.6-(round/100 * 0.4))

            agent.buffer.rewards = rewards_tensor
            agent.buffer.is_terminals = is_terminal
            agent.update()

    def save_models(self):
        """Save PPO params."""
        for param_name, agent in self.agents.items():
            torch.save(agent.policy_old.state_dict(), f"PPO_model_save_dir\\ppo_{param_name}.pth")