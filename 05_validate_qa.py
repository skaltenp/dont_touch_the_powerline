import os
import json
import glob
import random
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas()
import pandas as pd
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

def validate_qa(args, tokenizer, model, prompt):
    validation = ""
    with torch.no_grad():
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=4096,
            truncation=True
        )
        if args.device == "auto" and torch.cuda.is_available():
            inputs = inputs.to("cuda")
        validation = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.batch_decode(validation.detach().cpu().numpy())[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluator_llm_path", 
        default="LeoLM/leo-hessianai-70b-chat", 
        help="hf model path",
        type=str
    )
    parser.add_argument(
        "--validation_data_path", 
        default="benchmark/data.csv", 
        help="Path to qas in csv-format to validate",
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
        help="Device to run on",
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
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.evaluator_llm_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()


    question_groundedness_validation_prompt = """
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

    question_relevance_validation_prompt = """
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

    question_context_independence_validation_prompt = """
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

    answer_groundedness_validation_prompt = """
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

    answer_relevance_validation_prompt = """
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


    validation_documents = pd.read_csv(args.validation_data_path).to_dict("records")
    print(validation_documents)

    print("Generating feedback and score for each question and answer according to the validation criterions...")
    for context_qa in tqdm(validation_documents):
        question_groundedness_messages = [
            {"role": "user", "content": question_groundedness_validation_prompt.format(context=context_qa["context"], question=context_qa["question"])}
        ]
        question_relevance_messages = [
            {"role": "user", "content": question_relevance_validation_prompt.format(question=context_qa["question"])}
        ]
        question_context_independence_messages = [
            {"role": "user", "content": question_context_independence_validation_prompt.format(question=context_qa["question"])}
        ]

        answer_groundedness_messages = [
            {"role": "user", "content": question_groundedness_validation_prompt.format(context=context_qa["context"], question=context_qa["question"], answer=context_qa["answer"])}
        ]
        answer_relevance_messages = [
            {"role": "user", "content": question_relevance_validation_prompt.format(question=context_qa["question"], answer=context_qa["answer"])}
        ]
        
        validations = {
            "question_groundedness": validate_qa(
                args,
                tokenizer,
                model,
                tokenizer.apply_chat_template(
                    question_groundedness_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            "question_relevance": validate_qa(
                args,
                tokenizer,
                model,
                tokenizer.apply_chat_template(
                    question_relevance_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            "question_context_independence": validate_qa(
                args,
                tokenizer,
                model,
                tokenizer.apply_chat_template(
                    question_context_independence_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            "answer_groundedness": validate_qa(
                args,
                tokenizer,
                model,
                tokenizer.apply_chat_template(
                    answer_groundedness_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ),
            "answer_relevance": validate_qa(
                args,
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
            for validation_criterion, validation in validations.items():
                score = int(validation.split("Gesamtbewertung: ")[-1].split("<|im_end|>")[0].split("\n")[0].strip())
                feedback = validation.split("Gesamtbewertung: ")[-2].split("Bewertung: ")[1].split("<|im_end|>")[0].strip()
                context_qa.update(
                    {
                        f"{validation_criterion}_score": score,
                        f"{validation_criterion}_feedback": feedback,
                    }
                )
            try:
                data = pd.DataFrame(validation_documents)
                data_path = os.path.join("benchmark", "data_validated.csv")
                data.to_csv(data_path, index=False)
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
            continue