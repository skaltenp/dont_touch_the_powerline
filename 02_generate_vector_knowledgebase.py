import os
import random
import argparse

import numpy as np
from tqdm.auto import tqdm
import pandas as pd
tqdm.pandas()

import torch
from transformers import set_seed
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        default="documents.csv", 
        help="Path to csv with documents",
        type=str
    )
    parser.add_argument(
        "--knowledge_base_name",
        default="abb_switchgears", 
        help="Name of vector knowledgebase",
        type=str
    )
    parser.add_argument(
        "--embedding_model",
        default="intfloat/multilingual-e5-large", 
        help="Embedding model for vector knowledgebase",
        type=str
    )
    parser.add_argument(
        "--chunk_size", 
        default=512, 
        help="Length of the chunks in the vector database",
        type=int
    )
    parser.add_argument(
        "--random_seed", 
        default=42, 
        help="Random seed",
        type=int
    )
    parser.add_argument(
        "--device", 
        default="cuda", 
        help="Device to use for embeddings",
        type=str
    )
    args = parser.parse_args()
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    set_seed(args.random_seed)

    documents = pd.read_csv(args.csv_path)

    docs = [
        LangchainDocument(
            page_content=row["text"], 
            metadata={
                "source": row["source"]
            }
        ) for i, row in tqdm(documents.iterrows())
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=int(args.chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=[
            "-----",
            "\n# ", 
            "\n## ", 
            "\n### ", 
            "\n\n", 
            "\n",
        ],
    )

    docs_splitted = []
    for doc in docs:
        docs_splitted += text_splitter.split_documents([doc])

    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        multi_process=True,
        model_kwargs={"device": args.device},
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    index_name = f"{args.knowledge_base_name}_chunk_{args.chunk_size}_embeddings_{args.embedding_model.replace('/', '_')}"
    index_folder_path = f"indexes/{index_name}/"
    
    
    index = FAISS.from_documents(
        docs_splitted, 
        embedding_model, 
        distance_strategy=DistanceStrategy.COSINE
    )
    index.save_local(index_folder_path)
    
    docs_splitted_df = []
    for doc in docs_splitted:
        docs_splitted_df.append({
            "text": doc.page_content,
            "source": doc.metadata["source"]
        })
    docs_splitted_df = pd.DataFrame(docs_splitted_df, columns=["text", "source"])
    docs_splitted_csv_path = os.path.join(index_folder_path, f"documents_splitted.csv")
    docs_splitted_df.to_csv(docs_splitted_csv_path, index=False)


if __name__ == "__main__":
    main()