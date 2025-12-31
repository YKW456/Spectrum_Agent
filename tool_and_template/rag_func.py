import json

import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
pd.set_option('display.max_rows', None)


pd.set_option('display.max_columns', None)

pd.set_option('display.expand_frame_repr', False)

pd.set_option('display.max_colwidth', None)

def modify_recommended_materials(llm, material_selection_template, prompt, all_materials, recommended_materials):
    output = llm.invoke(input=material_selection_template.format
    (Requirement=prompt,Materials=all_materials,Recommended_Materials=recommended_materials)).content
    return output
def extract_materials(content):
    # Using re to match :Materials
    materials_matches = re.findall(r'Materials:\s*\[(.*?)\]', content)

    all_materials = []
    for match in materials_matches:
        materials = [m.strip() for m in match.split(',')]
        all_materials.extend(materials)

    # Find the unique materials.
    unique_materials = list(set(all_materials))

    return unique_materials
def search_materials(csv_file_path, ActionInput, search_num):
    print(search_num)
    df = pd.read_csv(csv_file_path, encoding_errors='ignore')

    docs = []
    for index, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        doc = Document(page_content=content, metadata={"row_index": index})
        docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        separators=["\n"],
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(docs)
    return_content = ""
    ActionInput = json.loads(ActionInput)
    print(ActionInput)
    exact_matches = []
    for query in ActionInput['keywords']:
        for doc in split_docs:
            if query in doc.page_content:
                exact_matches.append(doc)
        embedding = HuggingFaceEmbeddings(model_name=r'./Model/allminiv2/all-MiniLM-L6-v2')

        if exact_matches:
            search_num = min(search_num,len(exact_matches))
            print("Find exactly matching documents and perform semantic similarity filtering.")

            db = FAISS.from_documents(exact_matches, embedding)
            result_simi = db.similarity_search(query, k=min(search_num, len(exact_matches)))

        else:
            print("No exact matching document found, return the top N semantic similarity results.")
            db = FAISS.from_documents(split_docs, embedding)
            result_simi = db.similarity_search(query, k=search_num)

        return_content += "\n\n".join([doc.page_content for doc in result_simi])

    return return_content
