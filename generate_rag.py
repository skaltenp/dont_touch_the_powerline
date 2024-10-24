import os
import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import argparse

import torch
from transformers import set_seed
from rag_architecture import RAGArchitecture

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knowledgebase_path", 
        default="indexes/abb_switchgears_chunk_512_embeddings_intfloat_multilingual-e5-large", 
        help="hf dataset path",
        type=str
    )
    parser.add_argument(
        "--retriever_path", 
        default="intfloat/multilingual-e5-large", 
        help="hf model path",
        type=str
    )
    parser.add_argument(
        "--chunk_size", 
        default=512, 
        help="The chunk size",
        type=int
    )
    parser.add_argument(
        "--num_docs_retrieve", 
        default=5, 
        help="maximum documents",
        type=int
    )
    parser.add_argument(
        "--retriever_device", 
        default="cpu",
        help="Retriever device",
        type=str
    )
    parser.add_argument(
        "--generator_path", 
        default="DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental", 
        help="hf model path",
        type=str
    )
    parser.add_argument(
        "--generator_device", 
        default="auto",
        help="Generator device",
        type=str
    )
    parser.add_argument(
        "--assistant_tag", 
        default="<|im_start|>assistant", 
        help="assistant tag for the specific model",
        type=str
    )
    parser.add_argument(
        "--answer_end_tag", 
        default="<|im_end|>", 
        help="assistant tag for the specific model",
        type=str
    )
    parser.add_argument(
        "--benchmark_path", 
        default="benchmark/data_validated.csv",
        help="Benchmark path",
        type=str
    )
    parser.add_argument(
        "--random_seed", 
        default=42, 
        help="random seed",
        type=int
    )
    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    set_seed(args.random_seed)

    generated_questions = pd.read_csv(args.benchmark_path)

    print("Dataset before filtering:")
    print(
        generated_questions[
            [
                "question",
                "answer",
                "question_groundedness_score",
                "question_relevance_score",
                "question_context_independence_score",
                "answer_groundedness_score",
                "answer_relevance_score",
            ]
        ]
    )
    generated_questions = generated_questions.loc[
        (generated_questions["question_groundedness_score"] >= 0)
        & (generated_questions["question_relevance_score"] >= 0)
        & (generated_questions["question_context_independence_score"] >= 0)
        & (generated_questions["answer_groundedness_score"] >= 0)
        & (generated_questions["answer_relevance_score"] >= 0)
    ]
    print("Dataset after filtering:")
    print(
        generated_questions[
            [
                "question",
                "answer",
                "question_groundedness_score",
                "question_relevance_score",
                "question_context_independence_score",
                "answer_groundedness_score",
                "answer_relevance_score",
            ]
        ]
    )
    generated_questions = generated_questions.to_dict("records")

    if not os.path.exists("./results"):
        os.mkdir("./results")
   
    settings_name = f"chunk_{args.chunk_size}_retriever_{args.retriever_path.replace('/', '_')}_generator_{args.generator_path.split('/')[-1]}"
    output_file_name = f"./results/{settings_name}.json"

    try:
        with open(output_file_name, "r") as f:
            outputs = json.load(f)
    except Exception as e:
        outputs = []
    
    if (len(outputs) > 0 and len(outputs) > len([x for x in outputs if "generated_answer" in x])) or (len(outputs) == 0):
            rag = RAGArchitecture(
                knowledgebase_path=args.knowledgebase_path,
                retriever_path=args.retriever_path,
                retriever_device=args.retriever_device,
                num_docs_retrieve=args.num_docs_retrieve,
                generator_path=args.generator_path,
                generator_device=args.generator_device,
                assistant_tag=args.assistant_tag,
                answer_end_tag=args.answer_end_tag
            )

    for example in tqdm(generated_questions):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = rag.answer_with_retrieval(question)

        print("-" * 25)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f'Reference answer: {example["answer"]}')

        result = {
            "question": question,
            "true_answer": example["answer"],
            "source": example["source"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        outputs.append(result)

        with open(output_file_name, "w") as f:
            json.dump(outputs, f)