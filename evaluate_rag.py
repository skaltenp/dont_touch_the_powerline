import os
import random
import argparse
import json

import numpy as np
from tqdm.auto import tqdm
import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


def call_evaluator(tokenizer, model, prompt, args):
    outputs = ""
    with torch.no_grad():
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=4096,
            truncation=True
        )
        if args.device == "auto" and torch.cuda.is_available():
            inputs = inputs.to("cuda")
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
    res = tokenizer.batch_decode(outputs.detach().cpu().numpy())[0]
    feedback = res.split("###Feedback: ")[-1].split("[ERGEBNIS]")[0].split(args.answer_end_tag)[0].strip()
    score = res.split("###Feedback: ")[-1].split("[ERGEBNIS]")[-1].split(args.answer_end_tag)[0].strip()
    return feedback, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluator_llm_path", 
        default="LeoLM/leo-hessianai-70b-chat", 
        help="hf model path",
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
        "--generator_path", 
        default="DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental", 
        help="hf model path",
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
        "--random_seed", 
        default=42, 
        help="random seed",
        type=int
    )
    parser.add_argument(
        "--device", 
        default="auto", 
        help="Device for evaluator",
        type=str
    )
    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    set_seed(args.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.evaluator_llm_path
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.evaluator_llm_path,
        quantization_config=bnb_config,
        device_map=args.device,
        torch_dtype=torch.float16,
    )
    model.eval()

    evaluation_prompt_template = """
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

    settings_name = f"chunk_{args.chunk_size}_retriever_{args.retriever_path.replace('/', '_')}_generator_{args.generator_path.split('/')[-1]}"
    output_file_name = f"./results/{settings_name}.json"

    print(f"Running evaluation for {settings_name}:")
    
    answers = []
    if os.path.isfile(output_file_name):
        with open(output_file_name, "r") as j_file:
            answers = json.load(j_file)
    
    for experiment in tqdm(answers):
        if f"eval_score" in experiment:
            continue
        
        evaluation_prompt = evaluation_prompt_template.format(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        feedback, score = call_evaluator(tokenizer, model, evaluation_prompt, args)
        print(feedback, "|", score)
        experiment[f"eval_score"] = score
        experiment[f"eval_feedback"] = feedback
        
        with open(output_file_name, "w") as f:
            json.dump(answers, f)
    print("#" * 25)
    print("#" * 25)


if __name__ == "__main__":
    main()