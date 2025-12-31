from collections import defaultdict

import numpy as np
import optuna
from optuna.trial import TrialState




def calculate_material_conv(trials, penalty_factor=0.7):
    """
    Calculate the convergence/repeatability of material parameters to avoid searching near local optima
    Parameters:
    Trials: trial list
    Penalty_factor: Punishment factor for repeated combinations (0-1)
    return:
    Convergence score (0-1, lower indicates better diversity)
    """
    if len(trials) < 2:
        return 0.0  # 没有足够数据时返回最低收敛性

    # 获取所有材料参数名(假设前一半参数是材料)
    material_params = sorted([p for p in trials[0].params.keys()
                              if p.startswith('M') or p.lower().startswith('material')])

    # 统计每种材料组合出现的频率
    combo_counts = defaultdict(int)
    unique_combos = set()

    for trial in trials:
        if trial.state != TrialState.COMPLETE:
            continue

        # 获取当前试验的材料组合(按参数名排序确保顺序一致)
        materials = tuple(trial.params[p] for p in material_params)
        combo_counts[materials] += 1
        unique_combos.add(materials)

    num_trials = sum(combo_counts.values())
    if num_trials == 0:
        return 0.0

    # Calculate the penalty.
    repeat_penalty = 0.0
    for combo, count in combo_counts.items():
        if count > 1:
            repeat_penalty += (count - 1) ** 2

    max_possible_penalty = (len(trials) - len(unique_combos)) ** 2
    norm_penalty = repeat_penalty / max(max_possible_penalty, 1)

    # Calculate the diversity of material types
    material_variety = len(unique_combos) / len(trials)

    conv_score = 0.6 * (1 - norm_penalty) + 0.4 * material_variety  # 越高越好

    return min(conv_score, 1.0)


def calculate_depth_cov(trials, penalty_factor=0.5):
    """
    Calculate the coverage and repeat penalty of thickness parameters
    Parameters:
    Trials: trial list
    Penalty_factor: Punishment factor for repeated thickness combinations
    return:
    Convergence score (0-1, higher indicates better exploration)
    """
    if len(trials) < 2:
        return 0.0

    depth_params = sorted([p for p in trials[0].params.keys()
                           if p.startswith('D') or p.lower().startswith('depth')])

    depth_combo_counts = defaultdict(int)
    unique_depth_combos = set()
    depth_ranges = {p: {'min': float('inf'), 'max': float('-inf')} for p in depth_params}

    for trial in trials:
        if trial.state != TrialState.COMPLETE:
            continue

        depths = tuple(trial.params[p] for p in depth_params)
        depth_combo_counts[depths] += 1
        unique_depth_combos.add(depths)

        for p in depth_params:
            val = trial.params[p]
            depth_ranges[p]['min'] = min(depth_ranges[p]['min'], val)
            depth_ranges[p]['max'] = max(depth_ranges[p]['max'], val)

    num_depth_trials = sum(depth_combo_counts.values())
    if num_depth_trials == 0:
        return 0.0

    depth_repeat_penalty = 0.0
    for depths, count in depth_combo_counts.items():
        if count > 1:
            depth_repeat_penalty += (count - 1) ** 2

    max_depth_penalty = (len(trials) - len(unique_depth_combos)) ** 2
    norm_depth_penalty = depth_repeat_penalty / max(max_depth_penalty, 1)


    coverage_scores = []
    for p in depth_params:
        param_range = depth_ranges[p]['max'] - depth_ranges[p]['min']
        if param_range == 0:
            coverage_scores.append(0.0)  # 单一值，覆盖率最低
            continue

        values = [t.params[p] for t in trials if t.state == TrialState.COMPLETE]
        hist, _ = np.histogram(values, bins=10,
                               range=(depth_ranges[p]['min'], depth_ranges[p]['max']))
        coverage = np.sum(hist > 0) / 10  # 被覆盖的bin比例
        coverage_scores.append(coverage)

    avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
    print(avg_coverage)
    conv_score = 1 - (0.8 * norm_depth_penalty + 0.2 * (1 - avg_coverage))  # 越高越好

    return min(conv_score, 1.0)
def get_state(study):
    trials = study.trials
    recent_trials = trials[-20:]

    state = {
        'best_value': study.best_value,
        'recent_avg': np.mean([t.value for t in recent_trials]),
        'recent_std': np.std([t.value for t in recent_trials]),
        'stagnation': min((len(trials) - study.best_trial.number)/100 ,1),
        'material_diversity': calculate_material_conv(recent_trials),
        'depth_coverage': calculate_depth_cov(recent_trials),
    }
    print('state=',state)
    return state


def calculate_reward(old_best, new_best, state):
    improvement = max(0, old_best - new_best) if old_best is not None else 0
    reward = improvement * 10  # 放大改进信号
    reward += (1 - new_best)
    # Explore Reward.
    reward += (state['material_diversity'] * 0.2 * reward)
    reward += (state['depth_coverage'] * 0.05 * reward)

    # Punish Stagnate.
    if state['stagnation'] > 0.5:
        reward *= 0.5


    return np.clip(reward,0,1)

def plot_state():
    window_size = 20
    metrics = {
        'trials': [],
        'material_diversity': [],
        'depth_coverage': []
    }

    for i in range(20, len(study.trials)+1, 20):
        recent_trials = study.trials[max(0, i-window_size):i]
        metrics['trials'].append(i)
        metrics['material_diversity'].append(calculate_material_conv(recent_trials))
        metrics['depth_coverage'].append(calculate_depth_cov(recent_trials))
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))


    plt.subplot(1, 2, 1)
    plt.plot(metrics['trials'], metrics['material_diversity'], 'bo-')
    plt.title('Material Diversity Over Trials')
    plt.xlabel('Trial Number')
    plt.ylabel('Diversity Score (0-1)')
    plt.ylim(0, 1)
    plt.grid(True)


    plt.subplot(1, 2, 2)
    plt.plot(metrics['trials'], metrics['depth_coverage'], 'ro-')
    plt.title('Depth Coverage Over Trials')
    plt.xlabel('Trial Number')
    plt.ylabel('Coverage Score (0-1)')
    plt.ylim(0, 1)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\nStatistics:")
    print(f"Average material diversity: {np.mean(metrics['material_diversity']):.3f}")
    print(f"Average thickness coverage: {np.mean(metrics['depth_coverage']):.3f}")
    print(f"Optimal Material Diversity: {max(metrics['material_diversity']):.3f} (appears in round {metrics['trials'][np.argmax(metrics['material_diversity'])]} )")
    print(f"Optimal thickness coverage: {max(metrics['depth_coverage']):.3f} (appears in round {metrics['trials'][np.argmax(metrics['depth_coverage'])]} )")

