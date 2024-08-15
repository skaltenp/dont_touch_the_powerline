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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from ragatouille import RAGPretrainedModel
from langchain_core.language_models.llms import LLM

import plotly.express as px
import argparse


def call_llm(tokenizer, model, prompt):
    outputs = ""
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
    return tokenizer.batch_decode(outputs.detach().cpu().numpy())[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        default="LeoLM/leo-hessianai-70b-chat", 
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
        "--dataset_path", 
        default="skaltenp/open_manuals", 
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
        "--n_generations", 
        default=2500, 
        help="Number of initially generated QA combinations",
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

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()




    question_groundedness_critique_prompt = """
    Dir wird ein Kontext und eine Frage gegeben.
    Deine Aufgabe ist es, eine 'Gesamtbewertung' zu geben, die bewertet, wie gut man die gegebene Frage eindeutig mit dem gegebenen Kontext beantworten kann.
    Gib deine Antwort auf einer Skala von 1 bis 5 an, wobei 1 bedeutet, dass die Frage angesichts des Kontexts überhaupt nicht beantwortbar ist, und 5 bedeutet, dass die Frage klar und eindeutig mit dem Kontext beantwortbar ist.

    Gib deine Antwort wie folgt an:

    Antwort:::
    Bewertung: (deine Begründung für die Bewertung, als Text)
    Gesamtbewertung: (deine Bewertung, als Zahl zwischen 1 und 5)

    Du MUSST Werte für 'Bewertung:' und 'Gesamtbewertung:' in deiner Antwort angeben.

    Hier sind die Frage und der Kontext.

    Frage: {question}\n
    Kontext: {context}\n
    Antwort::: """

    question_relevance_critique_prompt = """
    Dir wird eine Frage gestellt.
    Deine Aufgabe ist es, eine 'Gesamtbewertung' zu geben, die darstellt, wie nützlich diese Frage für einen Techniker in der Wartung und Instandhaltung des Verteilnetz ist.
    Gib deine Antwort auf einer Skala von 1 bis 5 an, wobei 1 bedeutet, dass die Frage überhaupt nicht nützlich ist, und 5 bedeutet, dass die Frage extrem nützlich ist.

    Gib deine Antwort wie folgt an:

    Antwort:::
    Bewertung: (deine Begründung für die Bewertung, als Text)
    Gesamtbewertung: (deine Bewertung, als Zahl zwischen 1 und 5)

    Du MUSST Werte für 'Bewertung:' und 'Gesamtbewertung:' in deiner Antwort angeben.

    Hier ist die Frage.

    Frage: {question}\n
    Antwort::: """

    question_standalone_critique_prompt = """
    Dir wird eine Frage gestellt.
    Deine Aufgabe ist es, eine 'Gesamtbewertung' zu geben, die darstellt, wie kontextunabhängig diese Frage ist.
    Gib deine Antwort auf einer Skala von 1 bis 5 an, wobei 1 bedeutet, dass die Frage zusätzliche Informationen benötigt, um verstanden zu werden, und 5 bedeutet, dass die Frage für sich allein Sinn macht.
    Beispielsweise, wenn die Frage auf einen bestimmten Kontext verweist, wie 'im Kontext' oder 'im Dokument', muss die Bewertung 1 sein.
    Die Fragen können unklare technische Substantive oder Akronyme wie DIN, WWN, FNN enthalten und dennoch eine 5 sein: es muss lediglich für einen Bediener mit Zugang zur Dokumentation klar sein, worum es in der Frage geht.

    Zum Beispiel sollte "Wie viele Arten von Schaltern werden in Bild 44 erwähnt?" eine 1 erhalten, da es eine implizite Erwähnung eines Kontextes gibt, und somit die Frage nicht kontextunabhängig ist.

    Gib deine Antwort wie folgt an:

    Antwort:::
    Bewertung: (deine Begründung für die Bewertung, als Text)
    Gesamtbewertung: (deine Bewertung, als Zahl zwischen 1 und 5)

    Du MUSST Werte für 'Bewertung:' und 'Gesamtbewertung:' in deiner Antwort angeben.

    Hier ist die Frage.

    Frage: {question}\n
    Antwort::: """

    answer_groundedness_critique_prompt = """
    Dir wird ein Kontext, eine Frage und eine Antwort gegeben.
    Deine Aufgabe ist es, eine 'Gesamtbewertung' zu geben, die bewertet, wie gut man die gegebene Antwort auf dem Kontext basiert.
    Gib deine Antwort auf einer Skala von 1 bis 5 an, wobei 1 bedeutet, dass die Antwort gar nicht auf dem Kontext basiert, und 5 bedeutet, dass die Antwort klar und eindeutig auf dem Kontext basiert.

    Gib deine Antwort wie folgt an:

    Antwort:::
    Bewertung: (deine Begründung für die Bewertung, als Text)
    Gesamtbewertung: (deine Bewertung, als Zahl zwischen 1 und 5)

    Du MUSST Werte für 'Bewertung:' und 'Gesamtbewertung:' in deiner Antwort angeben.

    Hier sind die Frage, die Antwort und der Kontext.

    Frage: {question}\n
    Antwort: {answer}\n
    Kontext: {context}\n
    Antwort::: """

    answer_relevance_critique_prompt = """
    Dir wird eine Frage und eine Antwort gegeben.
    Deine Aufgabe ist es, eine 'Gesamtbewertung' zu geben, die darstellt, wie nützlich diese Antwort im Bezug auf die Frage für einen Techniker in der Wartung und Instandhaltung des Verteilnetz ist.
    Gib deine Antwort auf einer Skala von 1 bis 5 an, wobei 1 bedeutet, dass die Antwort NICHT relevant für die Frage ist, und 5 bedeutet, dass die Antwort relevant für die Frage ist.

    Gib deine Antwort wie folgt an:

    Antwort:::
    Bewertung: (deine Begründung für die Bewertung, als Text)
    Gesamtbewertung: (deine Bewertung, als Zahl zwischen 1 und 5)

    Du MUSST Werte für 'Bewertung:' und 'Gesamtbewertung:' in deiner Antwort angeben.

    Hier ist die Frage.

    Frage: {question}\n
    Antwort: {answer}\n
    Antwort::: """


    outputs = pd.read_csv("data.csv").to_dict("records")
    print(outputs)

    print("Generating critique for each QA couple...")
    for output in tqdm(outputs):
        question_groundedness_messages = [
            {"role": "user", "content": question_groundedness_critique_prompt.format(context=output["context"], question=output["question"])}
        ]
        question_relevance_messages = [
            {"role": "user", "content": question_relevance_critique_prompt.format(question=output["question"])}
        ]
        question_standalone_messages = [
            {"role": "user", "content": question_standalone_critique_prompt.format(question=output["question"])}
        ]

        answer_groundedness_messages = [
            {"role": "user", "content": question_groundedness_critique_prompt.format(context=output["context"], question=output["question"], answer=output["answer"])}
        ]
        answer_relevance_messages = [
            {"role": "user", "content": question_relevance_critique_prompt.format(question=output["question"], answer=output["answer"])}
        ]
        
        evaluations = {
            "question_groundedness": call_llm(
                tokenizer,
                model,
                tokenizer.apply_chat_template(
                    question_groundedness_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            "question_relevance": call_llm(
                tokenizer,
                model,
                tokenizer.apply_chat_template(
                    question_relevance_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            "question_standalone": call_llm(
                tokenizer,
                model,
                tokenizer.apply_chat_template(
                    question_standalone_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            "answer_groundedness": call_llm(
                tokenizer,
                model,
                tokenizer.apply_chat_template(
                    answer_groundedness_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            "answer_relevance": call_llm(
                tokenizer,
                model,
                tokenizer.apply_chat_template(
                    answer_relevance_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
        }
        try:
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    int(evaluation.split("Gesamtbewertung: ")[-1].split("<|im_end|>")[0].split("\n")[0].strip()),
                    evaluation.split("Gesamtbewertung: ")[-2].split("Bewertung: ")[1].split("<|im_end|>")[0].strip(),
                )
                output.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval,
                    }
                )
            try:
                data = pd.DataFrame(outputs)
                data.to_csv("data_assessed.csv", index=False)
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
            continue