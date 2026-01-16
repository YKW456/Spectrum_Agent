from bayes_config import Config



def search_materials(csv_file_path, query, search_num=3):
    Config.KEYWORDS = query
    Config.Tool_last_use = 0
    return "The materials required for the design have been updated."

