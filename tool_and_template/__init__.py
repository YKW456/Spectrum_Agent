import sys
sys.path.append('tool_and_template')
from typing import List, Union
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from .rag_func import search_materials

from .tool_function import *
from .prompt_template import *
from .general_template import *