# 贝叶斯优化的工作流定义
import random
import os
import pickle

import optuna.samplers

import optic_construction

import json
import csv

class RL_based_Bayes_workflow():
    def __init__(self,search_space,objective_func,start_trials, materials, range_min, range_max, config):
        self.start_trials = start_trials
        self.config = config
        self.storage_random = config.STORAGE_init
        self.storage = config.STORAGE

        self.search_space = search_space
        self.objective_func = objective_func
        self.temp_study = None
        self.history_study = None
        self.materials_history = set()
        self.layer_num = config.LAYERNUM
        self.path = config.PATH
        self.materials = materials
        self.range_min = range_min
        self.range_max = range_max
        self.color_target = config.COLOR_TARGET
        self.spectrum_target = config.SPECTRUM_TARGET
        self.material_weights = None
        self.max_repeat = 0
        self.history_recommendations = []
        self.last_material = None
        self.iteration_num = 0
        self.history_value = []
    def Sample_by_lhs_samplers(self, delete_old=True):
        if delete_old:
            folder_path = self.config.TARGET_DIR
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                # End with '.db' then delete.
                if os.path.isfile(file_path) and filename.endswith('.db'):
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
        self.temp_study = optuna.create_study(
            study_name='random',
            storage=self.storage_random,
            load_if_exists=False,  # This will overwrite existing study
            direction="minimize"
        )
        lhs_sampler = LatinHypercubeSampler(self.search_space, self.start_trials, seed = 42)
        self.temp_study.sampler = lhs_sampler

        self.temp_study.optimize(self.objective_func, n_trials=self.start_trials)
    def Sample_by_random_samplers(self, delete_old=True):
        if delete_old:
            folder_path = self.config.TARGET_DIR
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.db'):
                    os.remove(file_path)

        self.temp_study = optuna.create_study(
            study_name='random',
            storage=self.storage_random,
            load_if_exists=False,  # This will overwrite existing study
            direction="minimize" 
        )
        random_sampler = optuna.samplers.RandomSampler(seed = 42)
        self.temp_study.sampler = random_sampler
        self.temp_study.optimize(self.objective_func, n_trials=self.start_trials)

    def record_stage0(self):
        if self.temp_study is not None:
            all_trials = self.temp_study.trials
            # Get the best value at each epoch
            history = []
            best_so_far = float('inf')
            epoch = 0

            for trial in all_trials:
                if trial.value is not None:
                    epoch += 1
                    if trial.value < best_so_far:
                        best_so_far = trial.value
                    history.append((epoch, best_so_far))

            # Save history to file
            history_file = os.path.join(self.config.TARGET_DIR, 'search_history_0.csv')
            with open(history_file, 'w') as f:
                f.write("Epoch,Best_Value\n")
                for epoch, value in history:
                    f.write(f"{epoch},{value}\n")
    def record_stage1(self,name):
        history = []
        best_so_far = float('inf')
        epoch = 0

        for value in self.history_value:
            epoch += 1
            if value < best_so_far:
                best_so_far = value
            history.append((epoch, best_so_far))

        # Save history to file
        history_file = os.path.join(self.config.TARGET_DIR, 'search_history_1_{}.csv'.format(name))
        with open(history_file, 'w') as f:
            f.write("Epoch,Best_Value\n")
            for epoch, value in history:
                f.write(f"{epoch},{value}\n")
    def Update_for_optimization(self,name,delete_old):
        if delete_old:
            folder_path = self.config.TARGET_DIR
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.db'):
                    if not filename.endswith('stage0.db'):
                        os.remove(file_path)

        self.temp_study = optuna.load_study(
            study_name='random',
            storage=self.storage_random,
        )
        self.Update_trials_random(name)

    def Update_trials_random(self,name):
        if self.temp_study is not None:
            all_trials = self.temp_study.trials
            sorted_trials = sorted(all_trials,key=lambda t: (t.value is not None, t.value if t.value is not None else float('inf')))
            trials_to_keep = []
            for trial in sorted_trials:
                if len(trials_to_keep) >= 300:
                    break
                trials_to_keep.append(trial)
            new_study = optuna.create_study(
                study_name=name,
                storage=self.storage,
                load_if_exists=False,
                direction="minimize"
            )
            print("len of trials_to_keep = ",len(trials_to_keep))
            # Add filtered trials
            for trial in trials_to_keep:
                new_study.add_trial(trial)

            values = [t.value for t in new_study.trials if t.value is not None]
            if values:
                min_val = min(values)
                max_val = max(values)
                avg_all = sum(values) / len(values)

                # Calculate averages for different ranges
                first_100_avg = sum(values[:100]) / len(values[:100]) if len(values) >= 100 else None
                mid_100_avg = sum(values[100:200]) / len(values[100:200]) if len(values) >= 200 else None
                last_100_avg = sum(values[200:]) / len(values[200:]) if len(values) >= 100 else None

                print("\nValue Statistics:")
                print(f"Minimum value: {min_val}")
                print(f"Maximum value: {max_val}")
                print(f"Average of all values: {avg_all}")
                print(f"Average of first 100 values: {first_100_avg}")
                print(f"Average of middle 100 values (100-200): {mid_100_avg}")
                print(f"Average of last 100 values: {last_100_avg}")
            else:
                print("No valid values found in the study.")

            self.history_study = new_study
            self.temp_study = optuna.create_study(
                study_name='random',
                storage=self.storage,
                load_if_exists=False,
                direction="minimize"
            )

    def Update_trials(self,name):
        if self.temp_study is not None:
            all_trials = self.temp_study.trials + self.history_study.trials
            sorted_trials = sorted(all_trials, key=lambda t: (
            t.value is not None, t.value if t.value is not None else float('inf')))
            trials_to_keep = []
            materials_list = set()
            for trial in sorted_trials:
                if len(trials_to_keep) >= 300:
                    break
                trial_materials = tuple(
                    trial.params[f'M{i + 1}']
                    for i in range(self.layer_num)
                )
                if trial_materials not in materials_list:
                    materials_list.add(trial_materials)
                    trials_to_keep.append(trial)
            optuna.delete_study(study_name=name, storage=self.storage)
            optuna.delete_study(study_name='random', storage=self.storage)
            self.history_study = optuna.create_study(
                study_name=name,
                storage=self.storage,
                load_if_exists=False,
                direction="minimize"
            )
            for trial in trials_to_keep:
                self.history_study.add_trial(trial)
            self.temp_study = optuna.create_study(
                study_name='random',
                storage=self.storage,
                load_if_exists=False,
                direction="minimize"
            )
            values = [t.value for t in self.history_study.trials if t.value is not None]
            if values:
                min_val = min(values)
                max_val = max(values)
                avg_all = sum(values) / len(values)

                # Calculate averages for different ranges
                first_100_avg = sum(values[:100]) / len(values[:100]) if len(values) >= 100 else None
                mid_100_avg = sum(values[100:200]) / len(values[100:200]) if len(values) >= 200 else None
                last_100_avg = sum(values[200:]) / len(values[200:]) if len(values) >= 100 else None

                print("\nValue Statistics:")
                print(f"Minimum value: {min_val}")
                print(f"Maximum value: {max_val}")
                print(f"Average of all values: {avg_all}")
                print(f"Average of first 100 values: {first_100_avg}")
                print(f"Average of middle 100 values (100-200): {mid_100_avg}")
                print(f"Average of last 100 values: {last_100_avg}")
            else:
                print("No valid values found in the study.")
            self.iteration_num += 1


    def Update_history(self,name):
        # Select (candidate_num) best for the optimization
        self.Update_trials(name)
        candidate_num = 50
        history_trials = [t for t in self.history_study.trials]
        n_times = [0, candidate_num * 1, candidate_num * 3,candidate_num * 6]
        select_num = [candidate_num//3,candidate_num//3,candidate_num//3]
        selected_trials = []
        for j in range(3):
            segment = history_trials[n_times[j]:n_times[j + 1]]
            selected_trials.extend(random.sample(segment,select_num[j]) )

        # Add selected trials to temp_study.
        for trial in selected_trials:
            self.temp_study.add_trial(trial)

        self.max_repeat = 0
        self.history_recommendations = []

    def optimize_with_just_tpe_samplers(self):
        tpe_sampler = optuna.samplers.TPESampler(seed = 42)
        self.history_study.sampler = tpe_sampler
        for i in range(20):
            self.history_study.optimize(self.objective_func, n_trials=1)

            this_trial = self.history_study.trials[-1]
            self.history_value.append(this_trial.value)
    def optimize_with_just_random_samplers(self):
        random_sampler = optuna.samplers.RandomSampler(seed = 42)
        self.history_study.sampler = random_sampler
        for i in range(20):
            self.history_study.optimize(self.objective_func, n_trials=1)
            # Get the value for this trial
            this_trial = self.history_study.trials[-1]  # 获取最后一个完成的试验
            self.history_value.append(this_trial.value)



    def optimize_with_RL_plus_tpe_samplers(self, params_tpe):
        # Initialize
        tpe_sampler = optuna.samplers.TPESampler(
            **params_tpe
        )
        self.temp_study.sampler = tpe_sampler
        # current_material = tuple()

        # Use PPO to determine the hyperparameters of BO
        for i in range(20):
            repeat_count = 0
            self.temp_study.optimize(self.objective_func, n_trials=1)
            this_trial = self.temp_study.trials[-1]
            self.history_value.append(this_trial.value)
            # Extract first 'layer_nums' materials from params
            current_material = tuple(
                this_trial.params[f'M{i + 1}']
                for i in range(self.layer_num)
            )
            self.history_recommendations.append(current_material)
            for hist_values in self.history_recommendations:

                if current_material == hist_values:
                    repeat_count += 1
            self.max_repeat = max(self.max_repeat, repeat_count)
            print("max_repeat_count = ",self.max_repeat)
        if self.max_repeat > 10:
            return True
        else:
            return False


class RL_based_Bayes_workflow2():
    def __init__(self,search_times,range_min,range_max,config):
        self.search_times = search_times
        self.config = config
        self.storage = config.STORAGE
        self.storage_materials = config.STORAGE_MATERIALS
        self.storage_depth = config.STORAGE_DEPTHS
        self.color_target = config.COLOR_TARGET
        self.spectrum_target = config.SPECTRUM_TARGET

        self.path = config.PATH
        self.range_min = range_min
        self.range_max = range_max

        self.materials = []
        self.depth = []
        self.results = []
        self.best = []
    def record_stage2(self,study,num,name):
        if study is not None:
            all_trials = study.trials

            # Get the best value at each epoch
            history = []
            best_so_far = float('inf')
            epoch = 0

            for trial in all_trials:
                if trial.value is not None:
                    epoch += 1
                    if trial.value < best_so_far:
                        best_so_far = trial.value
                    history.append((epoch, best_so_far))

            # Save history to file
            history_file = os.path.join(self.config.TARGET_DIR, 'search_history_2_{}_{}.csv'.format(name,num))
            with open(history_file, 'w') as f:
                f.write("Epoch,Best_Value\n")
                for epoch, value in history:
                    f.write(f"{epoch},{value}\n")
    def search_top_n(self,n,name):
        print(self.storage)
        study = optuna.load_study(
            study_name=name,
            storage=self.storage,
        )

        sorted_trials = sorted(study.trials, key=lambda t: t.value)

        i = 0
        print('Start finding parameters for top n materials combinations.')
        while i < n:
            current_values = list(sorted_trials[i].params.values())
            half_len = len(current_values) // 2
            self.materials.append(current_values[:half_len])
            self.depth.append(current_values[half_len:])
            print('Materials are:',current_values[:half_len])
            print('Depths are',sorted_trials[i].value)
            i+=1


    def get_search_range(self,depth,search_interval):
        search_range = []
        print("depth= ",depth)
        for i in range(len(depth)):
            min_depth = max(0,depth[i] - 5) * search_interval
            max_depth = (depth[i] + 5) * search_interval

            search_range.append([min_depth,max_depth])
        return search_range
    def optimize_with_Sampler2(self,record,search_interval,name):
        tpe_sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.search_times//4,
            seed=42
        )

        # Create a study for each material combination
        for i, material_combo in enumerate(self.materials):
            study = optuna.create_study(
                direction="minimize"
            )

            search_range = self.get_search_range(self.depth[i],search_interval)
            print('self.depth[i] = {}'.format(self.depth[i]))
            print('search_range = {}'.format(search_range))

            def objective(trial):
                depths = [
                    trial.suggest_int(f"D{j + 1}", search_range[j][0], search_range[j][1])
                    for j in range(len(material_combo))
                ]
                # Create construction and calculate metrics
                materialLoader = optic_construction.MaterialLoader(self.path)


                mae = error = 0
                if self.spectrum_target != [[-1, -1, -1, -1, -1]]:
                    construction = optic_construction.Construction(
                        materialLoader,
                        self.range_min,
                        self.range_max,
                        material_combo,
                        depths,
                        interval=self.config.INTERVAL_SPECTRUM
                    )
                    A_list, R_list, T_list = construction.construct()
                    mae = construction.find_mae(self.spectrum_target,mode = self.config.MAE_MODE)

                if self.color_target != [-1, -1, -1]:
                    construction = optic_construction.Construction(
                        materialLoader,
                        400,
                        800,
                        material_combo,
                        depths,
                        interval=self.config.INTERVAL_COLOR
                    )
                    A_list, R_list, T_list = construction.construct()
                    rgb = construction.get_rgb()
                    error = optic_construction.get_color_error(rgb, self.color_target)

                result = mae + error / 50
                print(f'Material combo {i}: deltaE={error}, MAE={mae}')
                return result
            study.sampler = tpe_sampler
            study.optimize(objective, n_trials=self.search_times)
            if record:
                self.record_stage2(study,i,name)

            self.best.append(study.best_value)
            self.results.append(list(study.best_params.values()))

            print(f"Best result for material combo {material_combo}: {study.best_value}")

        with open(self.storage_materials, 'wb') as f:
            pickle.dump(self.materials, f)
        with open(self.storage_depth, 'wb') as f:
            pickle.dump(self.results, f)
