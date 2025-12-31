import json
import os
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from  tqdm import tqdm
# 初始化Sentence Transformer模型
model = SentenceTransformer(r'../Model/allminiv2')


def calculate_cosine_similarity(text1, text2):
    """计算两段文本的余弦相似度"""
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


def process_json_file(file_path):
    """处理单个JSON文件，计算平均余弦相似度"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    similarities = []
    for item in data:

        try:
            ref_answer = item['reference_answer']
            model_answer = item['model_answer']
            similarity = calculate_cosine_similarity(ref_answer, model_answer)
            similarities.append(similarity)
            print(item)
            print(similarity)
        except KeyError:
            continue

    if not similarities:  # 如果没有有效数据
        return None

    return {
        'file_name': os.path.basename(file_path),
        'average_similarity': np.mean(similarities),
        'num_samples': len(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'std_dev': np.std(similarities)
    }


def process_folder(folder_path, output_csv='similarity_results.csv'):
    """处理文件夹下的所有JSON文件"""
    folder_path = Path(folder_path)
    json_files = list(folder_path.glob('*.json'))

    results = []
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        result = process_json_file(json_file)
        if result is not None:
            results.append(result)

    # 转换为DataFrame并保存为CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"结果已保存到: {output_csv}")
    return df


# 使用示例
if __name__ == "__main__":
    folder_path = "qwen-NOV"  # 替换为你的文件夹路径
    output_csv = os.path.join(folder_path,"similarity_results.csv")  # 输出CSV文件名

    # 处理文件夹并保存结果
    results_df = process_folder(folder_path, output_csv)

    # 打印汇总统计
    if not results_df.empty:
        print("\n汇总统计:")
        print(f"平均相似度: {results_df['average_similarity'].mean():.4f}")
        print(f"总样本数: {results_df['num_samples'].sum()}")
        print(f"最高平均相似度文件: {results_df.loc[results_df['average_similarity'].idxmax(), 'file_name']}")
        print(f"最低平均相似度文件: {results_df.loc[results_df['average_similarity'].idxmin(), 'file_name']}")