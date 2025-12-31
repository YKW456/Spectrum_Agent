import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import color
from scipy.interpolate import Akima1DInterpolator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import illuminants
from tmm_core import (coh_tmm)
import numpy as np

class MaterialLoader():
    def __init__(self, path):
        self.path = path
        self.material = {}
        self.load_files()
        self.range_min = -1
        self.range_max = np.inf
        self.material_names = []
        self.flag = None
    def load_files(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                path = os.path.join(root, file)
                self.flag = path[-3:]
                if self.flag == 'csv':
                    df = pd.read_csv(path)
                    self.material[file[:-4]] = df
                else:
                    df = pd.read_excel(path)
                    self.material[file[:-5]] = df

    def find_range(self):
        for name in self.material:
            self.range_min = max(self.material[name].iloc[0, 0], self.range_min)
            self.range_max = min(self.material[name].iloc[-1, 0], self.range_max)
        print("min,max=", self.range_min, self.range_max)
        self.range_min *= 1000
        self.range_max *= 1000
        return self.range_min,self.range_max

    def get_material_names(self):
        self.material_names = []
        for name, nk in self.material.items():
            self.material_names.append(name)
        print(self.material_names)

        return self.material_names

    def plot_nk(self, which):
        combined_df = None

        for each in self.material_names:
            df = self.material[each]
            x = df.iloc[:, 0]  # wavelength
            y = df.iloc[:, 1] if which == 'n' else df.iloc[:, 2]  # select n or k

            if which == 'n':
                plt.plot(x, y, label='{}-n'.format(each))
            else:
                plt.plot(x, y, label='{}-k'.format(each))

            if combined_df is None:
                combined_df = pd.DataFrame({f'Wavelength': x, f'{each}_{which}': y})
            else:
                temp_df = pd.DataFrame({f'Wavelength': x, f'{each}_{which}': y})
                combined_df = pd.merge(combined_df, temp_df, on='Wavelength', how='outer')

        if combined_df is not None:
            combined_df.sort_values(by='Wavelength', inplace=True)
            combined_df.to_csv(f'Visualization\\all_materials_{which}_data.csv', index=False)

        plt.title(which, fontsize=26)
        plt.xlabel('wavelength of materials')
        plt.xlim(self.range_min / 1000, self.range_max / 1000)
        plt.legend()
        plt.savefig('Visualization\\materials_{}.png'.format(which))
        plt.close()
    def get_material_nk(self,name):
        return self.material[name]

class Construction():
    def __init__(self, mLoader, start, end, materials, depth_list, interval):
        self.mLoader = mLoader
        self.waveLength_start = start
        self.waveLength_end = end
        self.interval = interval
        self.lambda_list = np.linspace(start, end, self.interval)
        self.constrain_list = []
        self.materials = materials
        self.depth_list = [np.inf] + depth_list + [np.inf]
        self.n_list = [1]
        self.material_nk_fn = []
        self.num_layers = len(self.depth_list) - 2
        self.range_min = -1
        self.range_max = np.inf
        self.A_list = []
        self.R_list = []
        self.T_list = []
        self.mae_list = []

    def find_range(self):
        for name in self.materials:
            self.range_min = max(self.mLoader.material[name].iloc[0, 0],self.range_min)
            self.range_max = min(self.mLoader.material[name].iloc[-1, 0],self.range_max)
        print("min,max=",self.range_min*1000,self.range_max*1000)

    def find_mae(self,require,mode):
        total = 0
        # Example:
        # require = [[3000, 5000, 0,1, 'A'],[3000, 5010, 1,2, 'A']]
        for each in require:
            constrain = [int(self.interval*(each[0] - self.waveLength_start)/(self.waveLength_end - self.waveLength_start)),
                         int(self.interval*(each[1] - self.waveLength_start)/(self.waveLength_end - self.waveLength_start))]
            mae = 0
            if each[-1] =='A':
                LIST = self.A_list
            elif each[-1] =='R':
                LIST = self.R_list
            else:
                LIST = self.T_list
            constrain_A = LIST[constrain[0]:constrain[1]]
            # constrain_A is the desired value.
            constrain_given = [each[2]] * (constrain[1] - constrain[0])
            if mode == 'average':
                weight = (each[1] - each[0]) / (self.waveLength_end - self.waveLength_start)
                total += weight
            else:
                weight = each[3]

            for i in range(len(constrain_A)):
                mae += (abs(constrain_A[i] - constrain_given[i]) * weight)
            # for i in range(len(constrain_A)):
                # if constrain_A[i]>constrain_given[i]:
                #     mae += 0
                # else:
                #     mae +=(abs(constrain_A[i] - constrain_given[i]) * each[3])
            mae = mae/len(constrain_A)
            self.mae_list.append(mae)
        if mode == 'average':
            return np.sum(self.mae_list)/total
        else:
            return np.average(self.mae_list)
    def find_mae_2(self,flag,interpolated_values,plot):
        if flag == 'A':
            LIST = self.A_list
        elif flag == 'R':
            LIST = self.R_list
        else:
            LIST = self.T_list
        mae = 0
        for i in range(len(LIST)):
            mae += abs(LIST[i] - interpolated_values[i])
        if plot:
            plt.plot(self.lambda_list, interpolated_values)
            plt.ylim(0,1)
            plt.show()
        return mae

    def construct(self):

        for material in self.materials:
            try:
                df = self.mLoader.get_material_nk(material)
            except Exception as e:
                raise Exception(f"An error occurred while trying to get material nk data: {str(material)}")
            wl = df.iloc[:, 0]
            n = df.iloc[:, 1]
            k = df.iloc[:, 2]
            fun = lambda x: x * 1000
            wl = wl.apply(fun)
            wl = np.array(wl)
            complex_series = pd.Series(n + 1j * k)
            material_nk_data = np.column_stack((wl, complex_series))

            interp1d_nk = interp1d(material_nk_data[:, 0].real, material_nk_data[:, 1], kind='linear')
            self.material_nk_fn.append(interp1d_nk)


        # Dynamically generate a refractive index list based on the number of layers.
        for lambda_vac in self.lambda_list:
            self.n_list = [1]
            # 根据层数动态生成折射率列表
            for _ in range(self.num_layers):
                self.n_list.append(self.material_nk_fn[_](lambda_vac))
            self.n_list.append(1)
            # print(n_list)
            tmm_result = coh_tmm('s', self.n_list, self.depth_list, 0, lambda_vac)
            self.T_list.append(tmm_result['T'])
            self.R_list.append(tmm_result['R'])

        self.A_list = [1 - t - r for t, r in zip(self.T_list, self.R_list)]
        return self.A_list,self.R_list,self.T_list

    def plot(self, save_dir, name, target, mae):
        # Plotting code (unchanged)
        fig, ax = plt.subplots()
        ax.plot(self.lambda_list, self.T_list, 'b-', label='T')
        ax.plot(self.lambda_list, self.R_list, 'r-', label='R')
        ax.plot(self.lambda_list, self.A_list, 'g-', label='A')

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Value')
        ax.legend()
        plt.title('target={},mae={:.3f}%'.format(target, mae * 100))

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save plot
        save_path = os.path.join(save_dir, f"{name}_ART.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        # Save to CSV
        csv_path = os.path.join(save_dir, f"{name}_ART.csv")
        data = {
            'Wavelength (nm)': self.lambda_list,
            'Absorption': self.A_list,
            'Reflection': self.R_list,
            'Transmission': self.T_list
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
    def on_Click_Interpolate(self,method,min,max):
        global lambda_list, ART_list,last
        last = self.waveLength_start
        def onclick(event):
            global lambda_list, ART_list,last
            ix, iy = event.xdata, event.ydata
            if ix and ix > last and ix < self.waveLength_end:
                lambda_list.append(ix)
                ART_list.append(iy)
                last = ix

                line.set_data(lambda_list, ART_list)
                ax.draw_artist(line)
                fig.canvas.blit(ax.bbox)
            print('x =', ix)
            print('y =', iy)

        x = np.linspace(self.waveLength_start, self.waveLength_end, 1000)
        y = np.linspace(min, max, len(x))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        line, = ax.plot([], [], 'r-')  # Initialize an empty line object for drawing lines.
        ax.plot(x, y)

        lambda_list = []
        ART_list = []
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.grid(True)
        plt.show()
        lambda_list = [self.waveLength_start] + lambda_list + [self.waveLength_end]
        ART_list = [ART_list[0]] + ART_list + [ART_list[-1]]

        if method == "linear":
            interpolated_values = np.interp(self.lambda_list, lambda_list, ART_list)
        else:
            interp = Akima1DInterpolator(lambda_list, ART_list)
            interpolated_values = interp(self.lambda_list)

        return interpolated_values

    def plot_and_save_rgb(self, rgb_list, save_dir, name,color_target,error):
        """
        Directly display RGB colors and save them
        : param: rgc_list: RGB values in the format of [R, G, B] (range 0-1)
        : param: save_dir: Save directory
        : param: name: filename identifier
        """

        r, g, b = rgb_list[0], rgb_list[1], rgb_list[2]


        rgb_color = [[[r, g, b]]]  # 形状 (1, 1, 3)


        plt.figure(figsize=(4, 4))


        plt.imshow(rgb_color)

        plt.axis('off')
        plt.title(f"RGB: ({r}, {g}, {b}),target:{color_target},error{error:.3f}")

        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        save_path = os.path.join(save_dir, f"{name}_rgb_color.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"RGB color saved to: {save_path}")
    def get_rgb(self):
        air_n_fn = lambda wavelength: 1

        n_fn_list = [air_n_fn] + self.material_nk_fn + [air_n_fn]

        th_0 = 0
        reflectances = color.calc_reflectances(n_fn_list, self.depth_list, th_0)
        # reflectances = color.calc_transmittances(n_fn_list, self.depth_list, th_0)


        illuminant = illuminants.get_illuminant_D65()

        spectrum = color.calc_spectrum(reflectances, illuminant)
        color_dict = color.calc_color(spectrum)

        return color_dict['irgb']
