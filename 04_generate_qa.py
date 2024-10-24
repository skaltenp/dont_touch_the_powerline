import os
import random
import argparse

import numpy as np
from tqdm.auto import tqdm
import pandas as pd
tqdm.pandas()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


def generate_qa(args, tokenizer, model, prompt):
    qa = ""
    with torch.no_grad():
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=4096,
            truncation=True
        )
        if args.device == "auto" and torch.cuda.is_available():
            inputs = inputs.to("cuda")
        qa = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.batch_decode(qa.detach().cpu().numpy())[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluator_llm_path",
        default="LeoLM/leo-hessianai-70b-chat", 
        help="HF model path for evaluator/generator of qa and rag",
        type=str
    )
    parser.add_argument(
        "--knowledge_base_index_path",
        default="indexes/abb_switchgears_chunk_512_embeddings_intfloat_multilingual-e5-large", 
        help="Name of vector knowledgebase",
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
        "--n_qa", 
        default=2500, 
        help="Number of initially generated QA combinations",
        type=int
    )
    parser.add_argument(
        "--device", 
        default="auto", 
        help="Device to use for evaluator llm",
        type=str
    )
    args = parser.parse_args()
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    set_seed(args.random_seed)

    docs_splitted_path = os.path.join(args.knowledge_base_index_path, "documents_splitted.csv")
    docs_splitted = pd.read_csv(docs_splitted_path)

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
        device_map=args.device,
        torch_dtype=torch.float16,
    )
    model.eval()
    
    # Prompt Templates adapted from https://huggingface.co/learn/cookbook/rag_evaluation
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

    print(f"Generating {args.n_qa} questions and answers ...")
    qa_samples = []
    document_sample = random.sample(docs_splitted.to_dict("records"), min(len(docs_splitted), args.n_qa))
    for sampled_context in tqdm(document_sample):
        messages = [
            {
                "role": "system", 
                "content": system_message
            },
            {
                "role": "user", 
                "content": user_message.format(context=sampled_context["text"])
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        output_qa_couple = generate_qa(
            args,
            tokenizer,
            model,
            prompt
        )
        
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
        )[0].strip()

        print("Question:", question)
        print("Answer:", answer)
        print("#" * 25)
        

        qa_samples.append(
            {
                "source": sampled_context["source"],
                "context": sampled_context["text"],
                "question": question,
                "answer": answer,
            }
        )
        try:
            if not os.path.isdir("benchmark"):
                os.mkdir("benchmark")
            data = pd.DataFrame(qa_samples)
            data_path = os.path.join("benchmark", "data.csv")
            data.to_csv(data_path, index=False)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()