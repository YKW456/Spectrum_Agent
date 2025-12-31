import json
import os
from json import JSONDecodeError
from typing import List, Union,Optional
from bayes_config import Config
import rag_func
from langchain.tools import BaseTool


class Search_Materials_Tool(BaseTool):
    name: str = "Search_Materials_Tool"  # 添加类型注解
    description: str ='''
    Search for materials information using RAG (Retrieval-Augmented Generation).
    Args:
        keywords: The search term to look for in materials data
    Returns:
        String containing the search results
    '''
    def _run(self, keywords: str) -> str:
        """执行搜索并返回格式化的结果"""
        results = self.search(keywords)
        return results

    def search(self,keywords: str) -> str:
        """调用博查API进行搜索"""
        csv_file_path = 'Material_base/extracted_data_cleaned.csv'
        return rag_func.search_materials(csv_file_path, keywords)
class Update_Config(BaseTool):
    name: str = "Update_Config"  # 添加类型注解
    description: str = '''
    Update configuration parameters in Config class and regenerate dependent variables.

    Args:
        depth_max: Maximum depth value (positive integer, typically 100-1000), for example 200
        layer_num: Number of layers (positive integer, typically 1-24), for example 8
        search_interval: Search interval value (positive integer, typically 5-50), for example 20
        search_epochs: The total number of epoch (positive integer, typically 1000-10000), for example 5000
        color_target: RGB color target as list of 3 integers (0-255), for example [200,0,0]
        spectrum_target: Spectrum target as list of lists with format:
            [[start_wavelength, end_wavelength, value, weight, mode], ...]
        Where:
        - start_wavelength: Start of target wavelength range
        - end_wavelength: End of target wavelength range
        - value: Ideal spectral target value
        - weight: Importance weight for this range (higher = more important)
        - mode: Spectral type:
            'A': Absorption
            'R': Reflection
            'T': Transmission
    Returns:
        String confirmation of updated parameters
    '''

    def _run(self, tool_input: str, **kwargs) -> str:
        """处理可能包含非标准JSON的输入"""
        try:
            # 先尝试标准解析
            params = json.loads(tool_input)
        except JSONDecodeError:
            try:
                # 尝试替换单引号为双引号
                fixed_input = tool_input.replace("'", '"')
                params = json.loads(fixed_input)
            except Exception as e:
                return f"JSON解析失败: {str(e)}"

        return self.cfg_update(
            depth_max=params.get("depth_max"),
            layer_num=params.get("layer_num"),
            search_interval=params.get("search_interval"),
            search_epochs=params.get("search_epochs"),
            color_target=params.get("color_target"),
            spectrum_target=params.get("spectrum_target")
        )

    def cfg_update(self, depth_max: Optional[int] = None,
                   layer_num: Optional[int] = None,
                   search_interval: Optional[int] = None,
                   search_epochs: Optional[int] = None,
                   color_target: Optional[List[int]] = None,
                   spectrum_target: Optional[List[List[Union[int, str]]]] = None
                   ) -> str:
        """执行搜索并返回格式化的结果"""
        updates = []

        # Validate and update DEPTH_MAX
        if depth_max is not None:
            if isinstance(depth_max, int) and depth_max > 0:
                Config.DEPTH_MAX = depth_max
                updates.append(f"DEPTH_MAX updated to {depth_max}")
            else:
                updates.append(f"Invalid DEPTH_MAX value: {depth_max} (must be positive integer)")

        # Validate and update LAYERNUM
        if layer_num is not None:
            if isinstance(layer_num, int) and 1 <= layer_num <= 32:
                Config.LAYERNUM = layer_num
                updates.append(f"LAYERNUM updated to {layer_num}")
            else:
                updates.append(f"Invalid LAYERNUM value: {layer_num} (must be integer between 1-32)")

        # Validate and update SEARCH_INTERVAL
        if search_interval is not None:
            if isinstance(search_interval, int):
                Config.SEARCH_INTERVAL = min(search_interval, 100)
                updates.append(f"SEARCH_INTERVAL updated to {search_interval}")
            else:
                updates.append(f"Invalid SEARCH_INTERVAL value: {search_interval} (must be integer between 5-50)")
        if search_epochs is not None:
            if isinstance(search_epochs, int):
                Config.SEARCH_EPOCH = min(search_epochs, 100)
                updates.append(f"SEARCH_INTERVAL updated to {search_epochs}")
            else:
                updates.append(f"Invalid SEARCH_INTERVAL value: {search_epochs} (must be integer between 5-50)")
        # Validate and update COLOR_TARGET
        if color_target is not None:
            if (isinstance(color_target, list) and len(color_target) == 3 and
                    all(isinstance(x, int) and 0 <= x <= 255 for x in color_target)):
                Config.COLOR_TARGET = color_target
                updates.append(f"COLOR_TARGET updated to {color_target}")
            else:
                updates.append(f"Invalid COLOR_TARGET value: {color_target} (must be list of 3 integers 0-255)")

        # Validate and update SPECTRUM_TARGET
        if spectrum_target is not None:
            valid_spectrum = True
            if isinstance(spectrum_target, list):
                for entry in spectrum_target:
                    if not (isinstance(entry, list) and len(entry) == 5 and
                            all(isinstance(x, (int, float)) for x in entry[:4]) and  # 前四个元素可以是int或float
                            isinstance(entry[4], str) and entry[4] in ['T', 'R', 'A']):
                        valid_spectrum = False
                        break  # 修正：break应该在if not条件内

            if valid_spectrum:
                Config.SPECTRUM_TARGET = spectrum_target
                updates.append(f"SPECTRUM_TARGET updated to {spectrum_target}")
            else:
                updates.append("Invalid SPECTRUM_TARGET format. Each entry should be [num, num, num, num, 'T'/'R'/'A']")

        # Regenerate dependent variables if color or spectrum targets changed
        if color_target is not None or spectrum_target is not None:
            # Rebuild TARGET_DIR string
            target_dir = ''
            for every in Config.COLOR_TARGET:
                target_dir += str(every) + ' '
            for each in Config.SPECTRUM_TARGET:
                for every in each:
                    target_dir += str(every) + ','

            # Update all dependent paths
            Config.TARGET_DIR = os.path.join('../Visualization', target_dir)
            os.makedirs(Config.TARGET_DIR, exist_ok=True)
            Config.STORAGE = f'sqlite:///{Config.TARGET_DIR}/stage1.db'
            Config.STORAGE_MATERIALS = os.path.join(Config.TARGET_DIR, 'materials') + '.pkl'
            Config.STORAGE_DEPTHS = os.path.join(Config.TARGET_DIR, 'depths') + '.pkl'
            Config.Tool_last_use = 1
            updates.append(f"Regenerated dependent paths based on new targets")

        return "\n".join(updates) if updates else "No parameters were updated"




#
# def search_materials_tool(keyword: str, search_num: int = 3) -> str:
#     '''
#     Search for materials information using RAG (Retrieval-Augmented Generation).
#
#     Args:
#         keyword: The search term to look for in materials data
#         search_num: Number of results to return (default is 3)
#
#     Returns:
#         String containing the search results
#     '''
#     csv_file_path = 'Material_base/extracted_data_cleaned.csv'
#     return rag_tool.search_materials(csv_file_path, keyword, search_num=search_num)
# def update_config(
#         depth_max: Optional[int] = None,
#         layer_num: Optional[int] = None,
#         search_interval: Optional[int] = None,
#         color_target: Optional[List[int]] = None,
#         spectrum_target: Optional[List[List[Union[int, str]]]] = None
# ) -> str:
#     '''
#     Update configuration parameters in Config class and regenerate dependent variables.
#
#     Args:
#         depth_max: Maximum depth value (positive integer, typically 100-1000), for example 200
#         layer_num: Number of layers (positive integer, typically 1-24), for example 8
#         search_interval: Search interval value (positive integer, typically 5-50), for example 20
#         color_target: RGB color target as list of 3 integers (0-255), for example [200,0,0]
#         spectrum_target: Spectrum target as list of lists with format:
#             [[start_wavelength, end_wavelength, value, weight, mode], ...]
#         Where:
#         - start_wavelength: Start of target wavelength range
#         - end_wavelength: End of target wavelength range
#         - value: Ideal spectral target value
#         - weight: Importance weight for this range (higher = more important)
#         - mode: Spectral type:
#             'A': Absorption
#             'R': Reflectance
#             'T': Transmittance
#     Returns:
#         String confirmation of updated parameters
#     '''
#     updates = []
#
#     # Validate and update DEPTH_MAX
#     if depth_max is not None:
#         if isinstance(depth_max, int) and depth_max > 0:
#             Config.DEPTH_MAX = depth_max
#             updates.append(f"DEPTH_MAX updated to {depth_max}")
#         else:
#             updates.append(f"Invalid DEPTH_MAX value: {depth_max} (must be positive integer)")
#
#     # Validate and update layer_num
#     if layer_num is not None:
#         if isinstance(layer_num, int) and 1 <= layer_num <= 32:
#             Config.LAYERNUM = layer_num
#             updates.append(f"LAYERNUM updated to {layer_num}")
#         else:
#             updates.append(f"Invalid LAYERNUM value: {layer_num} (must be integer between 1-32)")
#
#     # Validate and update SEARCH_INTERVAL
#     if search_interval is not None:
#         if isinstance(search_interval, int):
#             Config.SEARCH_INTERVAL = min(search_interval,100)
#             updates.append(f"SEARCH_INTERVAL updated to {search_interval}")
#         else:
#             updates.append(f"Invalid SEARCH_INTERVAL value: {search_interval} (must be integer between 5-50)")
#
#     # Validate and update COLOR_TARGET
#     if color_target is not None:
#         if (isinstance(color_target, list) and len(color_target) == 3 and
#                 all(isinstance(x, int) and 0 <= x <= 255 for x in color_target)):
#             Config.COLOR_TARGET = color_target
#             updates.append(f"COLOR_TARGET updated to {color_target}")
#         else:
#             updates.append(f"Invalid COLOR_TARGET value: {color_target} (must be list of 3 integers 0-255)")
#
#     # Validate and update SPECTRUM_TARGET
#     if spectrum_target is not None:
#         valid_spectrum = True
#         if isinstance(spectrum_target, list):
#             for entry in spectrum_target:
#                 if not (isinstance(entry, list) and len(entry) == 5 and
#                         all(isinstance(x, (int, float)) for x in entry[:4]) and  # 前四个元素可以是int或float
#                         isinstance(entry[4], str) and entry[4] in ['T', 'R', 'A']):
#                     valid_spectrum = False
#                     break  # 修正：break应该在if not条件内
#
#         if valid_spectrum:
#             Config.SPECTRUM_TARGET = spectrum_target
#             updates.append(f"SPECTRUM_TARGET updated to {spectrum_target}")
#         else:
#             updates.append("Invalid SPECTRUM_TARGET format. Each entry should be [num, num, num, num, 'T'/'R'/'A']")
#
#     # Regenerate dependent variables if color or spectrum targets changed
#     if color_target is not None or spectrum_target is not None:
#         # Rebuild TARGET_DIR string
#         target_dir = ''
#         for every in Config.COLOR_TARGET:
#             target_dir += str(every) + ' '
#         for each in Config.SPECTRUM_TARGET:
#             for every in each:
#                 target_dir += str(every) + ','
#
#         # Update all dependent paths
#         Config.TARGET_DIR = os.path.join('../Visualization', target_dir)
#         os.makedirs(Config.TARGET_DIR, exist_ok=True)
#         Config.STORAGE = f'sqlite:///{Config.TARGET_DIR}/stage1.db'
#         Config.STORAGE_MATERIALS = os.path.join(Config.TARGET_DIR, 'materials') + '.pkl'
#         Config.STORAGE_DEPTHS = os.path.join(Config.TARGET_DIR, 'depths') + '.pkl'
#
#         updates.append(f"Regenerated dependent paths based on new targets")
#
#     return "\n".join(updates) if updates else "No parameters were updated"


# Example usage:
if __name__ == "__main__":
    # Print current config
    print("Current Config:")
    print(f"DEPTH_MAX: {Config.DEPTH_MAX}")
    print(f"LAYERNUM: {Config.LAYERNUM}")
    print(f"SEARCH_INTERVAL: {Config.SEARCH_INTERVAL}")
    print(f"COLOR_TARGET: {Config.COLOR_TARGET}")
    print(f"SPECTRUM_TARGET: {Config.SPECTRUM_TARGET}")
    print(f"TARGET_DIR: {Config.TARGET_DIR}")
    print(f"STORAGE: {Config.STORAGE}")
    print(f"STORAGE_MATERIALS: {Config.STORAGE_MATERIALS}")
    print(f"STORAGE_DEPTHS: {Config.STORAGE_DEPTHS}")

    # Update configuration
    print("\nUpdating configuration...")
    change = [None,None,]
    result = update_config(
        depth_max=400,
        layer_num=8,
        color_target=[128, 132, 251],
        spectrum_target=[
            [400, 525, 1, 1, 'T'],
            [525, 575, 1, 1, 'R'],
            [575, 1100, 1, 1, 'T']
        ]
    )
    print(result)

    # Print updated config
    print("\nUpdated Config:")
    print(f"DEPTH_MAX: {Config.DEPTH_MAX}")
    print(f"LAYERNUM: {Config.LAYERNUM}")
    print(f"SEARCH_INTERVAL: {Config.SEARCH_INTERVAL}")
    print(f"COLOR_TARGET: {Config.COLOR_TARGET}")
    print(f"SPECTRUM_TARGET: {Config.SPECTRUM_TARGET}")
    print(f"TARGET_DIR: {Config.TARGET_DIR}")
    print(f"STORAGE: {Config.STORAGE}")
    print(f"STORAGE_MATERIALS: {Config.STORAGE_MATERIALS}")
    print(f"STORAGE_DEPTHS: {Config.STORAGE_DEPTHS}")