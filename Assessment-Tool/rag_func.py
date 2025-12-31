from bayes_config import Config



def search_materials(csv_file_path, query, search_num=3):
    Config.KEYWORDS = query
    Config.Tool_last_use = 0
    return "The materials required for the design have been updated."

# csv_file_path = 'extracted_data_cleaned.csv'
# query = '红外'
# result = search_materials(csv_file_path, query, search_num=3)
# print(result)  # 输出为字符串（完全匹配的语义排序结果，或全局语义相似结果）
