import os
import random
import argparse
import json

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


def call_evaluator(tokenizer, model, prompt, args):
    outputs = ""
    with torch.no_grad():
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=4096,
            truncation=True
        ).to("cuda")
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
    res = tokenizer.batch_decode(outputs.detach().cpu().numpy())[0]
    feedback = res.split("###Feedback: ")[-1].split("[ERGEBNIS]")[0].split(args.answer_end_tag)[0].strip()
    result = res.split("###Feedback: ")[-1].split("[ERGEBNIS]")[-1].split(args.answer_end_tag)[0].strip()
    return feedback, result

def evaluate_answers(
    answer_path,
    tokenizer,
    model,
    evaluator_name,
    evaluation_prompt_template,
    args
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        evaluation_prompt = evaluation_prompt_template.format(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        feedback, score = call_evaluator(tokenizer, model, evaluation_prompt, args)
        print(feedback, "|", score)
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback
        
        with open(answer_path, "w") as f:
            json.dump(answers, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        default="LeoLM/leo-hessianai-70b-chat", 
        help="hf model path",
        type=str
    )
    parser.add_argument(
        "--model_to_evaluate_path", 
        default="", 
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
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    EVALUATION_PROMPT = """
###Aufgabenbeschreibung:
Eine Anweisung (kann eine Eingabe enthalten), eine zu bewertende Antwort, eine Referenzantwort, 
die eine Punktzahl von 5 erhält, und ein Bewertungsschema, das ein Bewertungskriterium darstellt, werden gegeben.

Schreiben Sie ein detailliertes Feedback, das die Qualität der Antwort streng basierend auf dem gegebenen Bewertungsschema 
bewertet, ohne allgemein zu bewerten.
Nach dem Schreiben eines Feedbacks geben Sie eine Punktzahl an, die eine ganze Zahl zwischen 1 und 5 ist. 
Sie sollten sich auf das Bewertungsschema beziehen.
Das Ausgabeformat sollte wie folgt aussehen: "Feedback: {{schreibe ein Feedback für die Kriterien}} [ERGEBNIS] 
{{eine ganze Zahl zwischen 1 und 5}}"
Bitte erzeugen Sie keine anderen Einleitungen, Abschlüsse und Erklärungen. Stellen Sie sicher, dass [ERGEBNIS] 
in Ihrer Ausgabe enthalten ist.
###Die zu bewertende Anweisung:
{instruction}

###Zu bewertende Antwort:
{response}

###Referenzantwort (Punktzahl 5):
{reference_answer}

###Bewertungskriterien:
[Ist die Antwort korrekt, genau und faktisch basierend auf der Referenzantwort?]
Punktzahl 1: Die Antwort ist völlig falsch, ungenau und/oder nicht faktisch.
Punktzahl 2: Die Antwort ist überwiegend falsch, ungenau und/oder nicht faktisch.
Punktzahl 3: Die Antwort ist teilweise korrekt, genau und/oder faktisch.
Punktzahl 4: Die Antwort ist größtenteils korrekt, genau und faktisch.
Punktzahl 5: Die Antwort ist vollständig korrekt, genau und faktisch.

###Feedback: """

    repo_ids = [args.model_to_evaluate_path, ]
    evaluator_name = model_name.split("/")[-1]
    for repo_id in repo_ids:
        evaluation_name = repo_id.split('/')[-1]
        for chunk_size in [512]:  # Add other chunk sizes (in tokens) as needed
            for embeddings in ["intfloat/multilingual-e5-large", ]:  # Add other embeddings as needed
                for rerank in [False, ]: #True 
                    settings_name = f"chunk_{chunk_size}_embeddings_{embeddings.replace('/', '_')}_rerank_{rerank}_reader-model_{evaluation_name}"
                    output_file_name = f"./output/rag_{settings_name}.json"
        
                    print(f"Running evaluation for {settings_name}:")
                    
                    print("Running evaluation...")
                    evaluate_answers(
                        output_file_name,
                        tokenizer,
                        model,
                        evaluator_name,
                        EVALUATION_PROMPT,
                        args
                    )
        print("#" * 25)
        print("#" * 25)



if __name__ == "__main__":
    main()