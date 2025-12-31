import optic_construction
import bayes_rl_model
from bayes_config import *

def objectiveMH(trial,path,materials_recommended,range_min, range_max,color_target,spectrum_target,n=8):
    materials_selected = []
    depths = []
    for i in range(n):
        materials_selected.append(trial.suggest_categorical(f"M{i + 1}", materials_recommended))
    for i in range(n):
        depths.append(trial.suggest_int(f"D{i + 1}", 0, Config.DEPTH_MAX // Config.SEARCH_INTERVAL) * Config.SEARCH_INTERVAL)

    materialLoader1 = optic_construction.MaterialLoader(path)
    mae = error = 0

    if spectrum_target !=[[-1,-1,-1,-1,-1]]:
        construction = optic_construction.Construction(materialLoader1, range_min, range_max, materials_selected, depths, interval=Config.INTERVAL_SPECTRUM)
        A_list, R_list, T_list = construction.construct()
        mae = construction.find_mae(spectrum_target, mode=Config.MAE_MODE)

    if color_target != [-1,-1,-1]:
        construction = optic_construction.Construction(materialLoader1, 400, 800, materials_selected, depths, interval=Config.INTERVAL_COLOR)
        A_list, R_list, T_list = construction.construct()
        rgb = construction.get_rgb()
        error = optic_construction.get_color_error(rgb, color_target)
    result = mae + error/50
    print('deltaE={},WMAE={}'.format(error,mae))
    return result




def runall():
    algorythm = 'PPO-BO'
    seed_everything(42)
    materialLoader = optic_construction.MaterialLoader(Config.PATH)
    materials = materialLoader.get_material_names()

    set1 = set(materials)
    set2 = set(Config.MATERIALS_RECOMMENDED)
    # Find the intersection
    materials_recommended = list(set1.intersection(set2))

    print(materials_recommended)
    Create_directory(Config,algorythm)

    materialLoader.material_names = materials_recommended
    search_space = {}
    for i in range(Config.LAYERNUM):
        search_space[f"M{i+1}"] = {"type": "categorical", "choices": materials_recommended}
        search_space[f"D{i+1}"] = {"type": "int", "low": 0, "high": Config.DEPTH_MAX // Config.SEARCH_INTERVAL}
    range_min = float('inf')
    range_max = float('-inf')

    for target in Config.SPECTRUM_TARGET:
        current_start, current_end = target[0], target[1]

        # Update the minimum.
        if current_start < range_min:
            range_min = current_start

        # Update the maximum.
        if current_end > range_max:
            range_max = current_end
    RL_based_BO = bayes_rl_model.RL_based_Bayes_workflow(search_space,
         lambda trial: objectiveMH(trial, Config.PATH, materials_recommended, range_min, range_max,
                                   Config.COLOR_TARGET, Config.SPECTRUM_TARGET, n=Config.LAYERNUM)
         , start_trials = 300, materials=materials_recommended, range_min=range_min,
         range_max=range_max, config=Config)

    # Stage1: Random search.
    RL_based_BO.Sample_by_random_samplers(delete_old=True)
    RL_based_BO.record_stage0()
    # Stage2: Algorythm search.
    RL_based_BO.Update_for_optimization(algorythm,delete_old=True)
    bayes_rl_model.run_RL_optimization(RL_based_BO, Config.SEARCH_EPOCH)
    # Stage3: Top n search.
    BO2 = bayes_rl_model.RL_based_Bayes_workflow2(search_times=50, range_min=range_min, range_max=range_max, config=Config)
    BO2.search_top_n(n=5,name=algorythm)
    BO2.optimize_with_Sampler2(record = True, search_interval=Config.SEARCH_INTERVAL,name=algorythm)
    # Stage4: Save results.
    materials, depths = load_2_items(Config.STORAGE_MATERIALS, Config.STORAGE_DEPTHS)
    for i in range(len(materials)):
        name = 'result' + algorythm + str(i)
        # spectrum_target = [[400,1100,0,1,'A']]
        if Config.SPECTRUM_TARGET != [[-1, -1, -1, -1, -1]]:
            construction = optic_construction.Construction(materialLoader, range_min, range_max, materials[i], depths[i], Config.INTERVAL_SPECTRUM)
            A_list, R_list, T_list = construction.construct()
            mae = construction.find_mae(Config.SPECTRUM_TARGET, mode=Config.MAE_MODE)
            construction.plot(Config.TARGET_DIR, name, Config.SPECTRUM_TARGET, mae)
        if Config.COLOR_TARGET != [-1, -1, -1]:
            construction = optic_construction.Construction(materialLoader, 400, 800, materials[i], depths[i], Config.INTERVAL_COLOR)
            A_list, R_list, T_list = construction.construct()
            rgb = construction.get_rgb()
            error = optic_construction.get_color_error(rgb, Config.COLOR_TARGET)
            construction.plot_and_save_rgb(rgb, Config.TARGET_DIR, name, Config.COLOR_TARGET, error)


if __name__  == '__main__':
    runall()