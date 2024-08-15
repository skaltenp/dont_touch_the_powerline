import os
import json
import glob
import random
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas()
import pandas as pd
from typing import Optional, List, Tuple

import torch
import datasets
from huggingface_hub import notebook_login, login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, pipeline

from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from ragatouille import RAGPretrainedModel
from langchain_core.language_models.llms import LLM

import plotly.express as px
import argparse

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: str,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of size `chunk_size` characters and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", ".", " ", ""],
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def load_embeddings(
    langchain_docs: List[LangchainDocument],
    chunk_size: int,
    embedding_model_name: Optional[str] = "intfloat/multilingual-e5-large",
) -> FAISS:
    """
    Creates a FAISS index from the given embedding model and documents. Loads the index directly if it already exists.

    Args:
        langchain_docs: list of documents
        chunk_size: size of the chunks to split the documents into
        embedding_model_name: name of the embedding model to use

    Returns:
        FAISS index
    """
    # load embedding_model
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        model_kwargs={"device": "cuda:0"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # set True to compute cosine similarity
    )

    # Check if embeddings already exist on disk
    index_name = (
        f"index_chunk_{chunk_size}_embeddings_{embedding_model_name.replace('/', '_')}"
    )
    index_folder_path = f"./data/indexes/{index_name}/"
    if os.path.isdir(index_folder_path):
        return FAISS.load_local(
            index_folder_path,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True
        )

    else:
        print("Index not found, generating it...")
        docs_processed = split_documents(
            chunk_size,
            langchain_docs,
            embedding_model_name,
        )
        knowledge_index = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        knowledge_index.save_local(index_folder_path)
        return knowledge_index


def call_rag_reader(tokenizer, model, prompt: str, gpt2=False):
    outputs = ""
    print(gpt2)
    if gpt2:
        prompt = prompt.replace("<|endoftext|>", "") + " Antwort:"
        res = model(prompt, max_new_tokens=50, return_full_text=False,)
        res = res[0]["generated_text"]
    else:
        with torch.no_grad():
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=4096,
                truncation=True
            ).to("cuda:0")
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        res = tokenizer.batch_decode(outputs.detach().cpu().numpy())[0]
        res = res.split(args.assistant_tag)[-1].split(args.answer_end_tag)[0].split("<|")[0]
        if res:
            res = res.strip()
    print("#" * 25)
    print(res)
    print("#" * 25)
    return res

def answer_with_rag(
    question: str,
    tokenizer,
    llm,
    knowledge_index: VectorStore,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
    gpt2=False
) -> Tuple[str, List[LangchainDocument]]:
    """Answer a question using RAG with the given knowledge index."""
    # Gather documents with retriever
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    # Optionally rerank results
    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtrahierte Dokumente:\n"
    context += "\n".join(
        [f"Dokument {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
    )
    
    content = "Kontext:\n{context}\n---\nHier ist nun die Frage, die Sie beantworten müssen.\n\nFrage: {question}"
    content = content.format(question=question, context=context)
    messages = [
        {"role": "system", "content": "Mit den Informationen aus dem Kontext geben Sie eine umfassende Antwort auf die Frage.\nAntworten Sie nur auf die gestellte Frage, Ihre Antwort sollte knapp und relevant zur Frage sein.\nGeben Sie die Nummer des Quelldokuments an, wenn dies relevant ist.\nWenn die Antwort nicht aus dem Kontext abgeleitet werden kann, geben Sie keine Antwort.",},
        {"role": "user", "content":content}
    ]    
    if gpt2:
        final_prompt = messages[0]["content"] + "\n" + messages[1]["content"]
    else:
        final_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            #return_tensors="pt"
        )

    # Redact an answer
    answer = call_rag_reader(tokenizer, llm, final_prompt, gpt2=gpt2)

    return answer, relevant_docs


def run_rag_tests(
    eval_dataset: datasets.Dataset,
    tokenizer,
    llm,
    knowledge_index: VectorStore,
    output_file: str,
    reranker: Optional[RAGPretrainedModel] = None,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
    gpt2=False,
    num_docs_final=5
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = answer_with_rag(
            question, tokenizer, llm, knowledge_index, reranker=reranker, gpt2=gpt2, num_docs_final=num_docs_final
        )
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_doc"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        default="DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental", 
        help="hf model path",
        type=str
    )
    parser.add_argument(
        "--embedding_model_path", 
        default="intfloat/multilingual-e5-large", 
        help="hf model path",
        type=str
    )
    parser.add_argument(
        "--use_fast_tokenizer", 
        default=True, 
        help="fast tokenizer usage", 
        type=bool
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
        "--index_dataset_path", 
        default="skaltenp/open_manuals", 
        help="hf dataset path",
        type=str
    )
    parser.add_argument(
        "--dataset_path", 
        default="data_assessed.csv", 
        help="hf dataset path",
        type=str
    )
    parser.add_argument(
        "--seq_length", 
        default=4096, 
        help="Sequence / context length of the llm",
        type=int
    )
    parser.add_argument(
        "--random_seed", 
        default=42, 
        help="random seed",
        type=int
    )
    parser.add_argument(
        "--num_docs_final", 
        default=5, 
        help="maximum documents",
        type=int
    )
    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    set_seed(args.random_seed)

    login()




    model_name = args.model_path
    dataset_path = args.dataset_path

    READER_MODEL_NAME = model_name.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    GPT2 = False

    if "gpt2" in model_name:
        GPT2 = True
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            #quantization_config=bnb_config,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        model = pipeline('text-generation', model=model,
                 tokenizer=tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        model.eval()

    generated_questions = pd.read_csv(dataset_path)

    print("Evaluation dataset before filtering:")
    print(
        generated_questions[
            [
                "question",
                "answer",
                "question_groundedness_score",
                "question_relevance_score",
                "question_standalone_score",
                "answer_groundedness_score",
                "answer_relevance_score",
            ]
        ]
    )
    generated_questions = generated_questions.loc[
        (generated_questions["question_groundedness_score"] >= 4)
        & (generated_questions["question_relevance_score"] >= 3)
        & (generated_questions["question_standalone_score"] >= 4)
        & (generated_questions["answer_groundedness_score"] >= 4)
        & (generated_questions["answer_relevance_score"] >= 3)
    ]
    print("============================================")
    print("Final evaluation dataset:")
    print(
        generated_questions[
            [
                "question",
                "answer",
                "question_groundedness_score",
                "question_relevance_score",
                "question_standalone_score",
                "answer_groundedness_score",
                "answer_relevance_score",
            ]
        ]
    )
    eval_dataset = datasets.Dataset.from_pandas(
        generated_questions, split="train", preserve_index=False
    )

    ds = datasets.load_dataset(args.index_dataset_path)["train"]
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in tqdm(ds)
    ]

    RAG_PROMPT_TEMPLATE = """
    Mit den Informationen aus dem Kontext geben Sie eine umfassende Antwort auf die Frage. 
    Antworten Sie nur auf die gestellte Frage, 
    Ihre Antwort sollte knapp und relevant zur Frage sein. 
    Geben Sie die Nummer des Quelldokuments an, wenn dies relevant ist. 
    Wenn die Antwort nicht aus dem Kontext abgeleitet werden kann, 
    geben Sie keine Antwort.

    Kontext:
    {context}
    ---
    Hier ist nun die Frage, die Sie beantworten müssen.

    Frage: {question}

    """

    messages = [
        {
            "role": "system", "content": "Mit den Informationen aus dem Kontext geben Sie eine umfassende Antwort auf die Frage.\nAntworten Sie nur auf die gestellte Frage, Ihre Antwort sollte knapp und relevant zur Frage sein.\nGeben Sie die Nummer des Quelldokuments an, wenn dies relevant ist.\nWenn die Antwort nicht aus dem Kontext abgeleitet werden kann, geben Sie keine Antwort.",
        },
        {
            "role": "user", "content":"Kontext:\n{context}\n---\nHier ist nun die Frage, die Sie beantworten müssen.\n\nFrage: {question}"
        }
    ]




    if not os.path.exists("./output"):
        os.mkdir("./output")

    for chunk_size in [512]:
        for embeddings in [args.embedding_model_path, ]:
            for rerank in [False, ]: 
                settings_name = f"chunk_{chunk_size}_embeddings_{embeddings.replace('/', '_')}_rerank_{rerank}_reader-model_{READER_MODEL_NAME}"
                output_file_name = f"./output/rag_{settings_name}.json"

                print(f"Running evaluation for {settings_name}:")

                print("Loading knowledge base embeddings...")
                knowledge_index = load_embeddings(
                    RAW_KNOWLEDGE_BASE,
                    chunk_size=chunk_size,
                    embedding_model_name=embeddings,
                )

                print("Running RAG...")
                reranker = (
                    RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
                    if rerank
                    else None
                )
                run_rag_tests(
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                    llm=model,
                    knowledge_index=knowledge_index,
                    output_file=output_file_name,
                    reranker=reranker,
                    verbose=False,
                    test_settings=settings_name,
                    gpt2=GPT2,
                    num_docs_final=args.num_docs_final
                )

