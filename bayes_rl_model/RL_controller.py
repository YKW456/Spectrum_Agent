import time

from RL_state_calc import *
from MultiPPO_agent import *

action_space_config = {

    'gamma_func': {'type': 'continuous', 'min': 0.1, 'max': 0.8},
    # The proportion of samples.
    'n_ei_candidates': {'type': 'continuous', 'min': 10, 'max': 50},
    # Number of candidate samples in the sampling stage.
    'prior_weight': {'type': 'continuous', 'min': 0.01, 'max': 1.0}
    # The weight of prior distribution is used to balance the weight relationship between newly sampled points and historical observation data. Prior weight has a stronger influence, more conservative algorithm, and stronger exploratory ability.
}

class RLForBayesOpt:
    def __init__(self, initial_params):
        self.params_history = [initial_params]
        self.state_history = []
        self.reward_history = []
        self.action_history = []
        self.is_terminal = []
        self.ppo_controller = MultiAgentPPO(action_space_config)

    def get_action(self, state):
        return self.ppo_controller.get_actions(state)
    def apply_action(self, action):
        params_tpe = {
            'n_startup_trials': 0,
            'gamma': lambda n: int(action['gamma_func'] * n),
            'n_ei_candidates': int(action['n_ei_candidates']),
            'prior_weight': action['prior_weight'],
            'weights':lambda n: np.ones(n),
            'seed': 42,
            'multivariate':True
        }
        return params_tpe
    def update_policy(self, state, action, reward, flag, round):
        """Update policy of PPO."""
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.is_terminal.append(flag)
        update_freq = 5

        if len(self.reward_history) % update_freq == 0 and len(self.reward_history):
            self.ppo_controller.update_agents(
                round,
                rewards=self.reward_history[-update_freq:],
                is_terminal = self.is_terminal
            )
            self.ppo_controller.save_models()

def run_basic_random_optimization(RL_based_BO, total_epoch):
    RL_based_BO.Update_trials(name='RANDOM')
    for i in range(total_epoch//20):
        RL_based_BO.optimize_with_just_random_samplers()
    RL_based_BO.record_stage1('RANDOM')
    return 0

def run_basic_bo_optimization(RL_based_BO, total_epoch):
    RL_based_BO.Update_trials(name='BASIC-BO')
    for i in range(total_epoch//20):
        RL_based_BO.optimize_with_just_tpe_samplers()
    RL_based_BO.record_stage1('BASIC-BO')
    return 0

def run_RL_optimization(RL_based_BO, max_iterations):
    current_params = {
        'gamma': lambda n: int(0.7 * n),
        'n_ei_candidates': 24,
        'prior_weight': 1.0,
        'seed': 42,
    }

    rl_controller = RLForBayesOpt(current_params)
    max_rounds = 10
    while 1:
        old_best = None
        RL_based_BO.Update_history('PPO-BO')
        for round in range(max_rounds):
            flag = RL_based_BO.optimize_with_RL_plus_tpe_samplers(current_params)
            # Calculating actions and reward.
            state = get_state(RL_based_BO.temp_study)
            reward = calculate_reward(
                old_best=old_best,
                new_best=state['best_value'],
                state=state
            )
            old_best = state['best_value']

            # Calculating new actions.
            action = rl_controller.get_action(list(state.values()))  # 将state转为list
            rl_controller.params_history.append(current_params)
            current_params = rl_controller.apply_action(action)
            # If you need to Train:
            # rl_controller.update_policy(
            #     state = list(state.values()),
            #     action = action,
            #     reward = reward,
            #     flag = flag,
            #     round = round
            # )

            print("Total Trial is {}, now is at Trial {}".format(max_iterations,len(RL_based_BO.history_value)))
            if len(RL_based_BO.history_value) >= max_iterations:
                RL_based_BO.Update_history('PPO-BO')
                RL_based_BO.record_stage1('PPO_BO')
                return 0

            print(f"\n=== Round {round + 1}/{max_rounds} ===")
            print(f"Best value: {state['best_value']:.4f}")
            print(f"Reward: {reward:.2f}")
            print("Action applied :")
            for k, v in action.items():
                print(f" {k}: {v}")
            if flag:
                break
