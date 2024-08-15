import os
import random
import argparse

import numpy as np
from tqdm.auto import tqdm
import pandas as pd

import torch
import datasets
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

import plotly.express as px


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


def main():
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
        default="data_generation/documents.csv", 
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

    generated_questions = pd.read_csv(dataset_path)
    ds = datasets.Dataset.from_pandas(
        generated_questions, split="train", preserve_index=False
    )["train"]

    langchain_docs = [
        LangchainDocument(
            page_content=doc["text"], 
            metadata={
                "source": doc["source"]
            }
        ) for doc in tqdm(ds)
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=200,
        add_start_index=True,
        separators=[
            "\n# ", 
            "\n## ", 
            "\n### ", 
            "\n\n", 
            ".", 
            " ", 
            ""
        ],
    )

    docs_processed = []
    for doc in langchain_docs:
        docs_processed += text_splitter.split_documents([doc])

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

    N_GENERATIONS = args.n_generations  # We intentionally generate 500 (close source) or 2500 (open source) QA couples here as many will not survive the validation

    print(f"Generating {N_GENERATIONS} QA couples...")
    system_message = """
    Deine Aufgabe ist es, eine faktische Frage und eine Antwort zu formulieren, basierend auf einem gegebenen Kontext.
    Deine faktische Frage sollte mit einem spezifischen, prägnanten Stück Fakteninformation aus dem Kontext beantwortbar sein.
    Deine faktische Frage sollte im selben Stil formuliert werden, wie Fragen, die Benutzer in einer Suchmaschine stellen könnten.
    Deine faktische Frage darf NICHT auf den Kontext verweisen.
    Phrasen wie "laut des Textes", "wie im Kontext", oder eine Erwähnung des Kontexts sind verboten."""
    
    user_message = """
    Gib deine Antwort wie folgt an:
    
    Ausgabe:::
    Faktische Frage: (deine faktische Frage)
    Antwort: (deine Antwort auf die faktische Frage)
    
    Hier ist der Kontext.
    
    Kontext: {context}\n
    Ausgabe::: """
    outputs = []
    for sampled_context in tqdm(random.sample(docs_processed, N_GENERATIONS)):
        messages = [
            {
                "role": "system", 
                "content": system_message
            },
            {
                "role": "user", 
                "content": user_message.format(context=sampled_context.page_content)
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        output_qa_couple = call_llm(
            tokenizer,
            model,
            prompt
        )

        try:
            question = output_qa_couple.split(
                args.assistant_tag
            )[-1].split(
                "Faktische Frage: "
            )[-1].split(
                "Antwort: "
            )[0].split(
                args.answer_end_tag
            )[0].strip()
            
            answer = output_qa_couple.split(
                args.assistant_tag
            )[-1].split(
                "Antwort: "
            )[-1].split(
                args.answer_end_tag
            )[0]

            print("Question:", question)
            print("Answer:", answer)
            print("#" * 25)
            
            assert len(answer) <= 512, "Answer is too long"
            outputs.append(
                {
                    "context": sampled_context.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": sampled_context.metadata["source"],
                }
            )
            try:
                data = pd.DataFrame(outputs)
                data.to_csv("data.csv", index=False)
            except Exception as e:
                print(e)
        except:
            continue


if __name__ == "__main__":
    main()