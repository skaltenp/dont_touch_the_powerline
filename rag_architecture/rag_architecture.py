import os
import random
import argparse

import numpy as np
from tqdm.auto import tqdm
import pandas as pd
tqdm.pandas()

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS


class RAGArchitecture():

    def __init__(
            self,
            knowledgebase_path="",
            retriever_path="",
            retriever_device="",
            num_docs_retrieve=5,
            generator_path="",
            generator_device="",
            assistant_tag="",
            answer_end_tag="",
        ):
        self.knowledgebase_path=knowledgebase_path
        self.retriever_path=retriever_path
        self.retriever_device=retriever_device
        self.num_docs_retrieve=num_docs_retrieve
        self.generator_path=generator_path
        self.generator_device=generator_device
        self.assistant_tag=assistant_tag
        self.answer_end_tag=answer_end_tag

        self.retriever = self.load_retriever()
        self.tokenizer, self.generator, self.gpt2 = self.load_generator()

    def load_retriever(self):
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.retriever_path,
            multi_process=True,
            model_kwargs={"device": self.retriever_device},
            encode_kwargs={
                "normalize_embeddings": True
            },
        )    
        retriever = FAISS.load_local(
            self.knowledgebase_path,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True
        )
        return retriever
    
    def load_generator(self):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.generator_path
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        gpt2 = False

        if "gpt2" in self.generator_path:
            gpt2 = True
            model = AutoModelForCausalLM.from_pretrained(
                self.generator_path,
                device_map=self.generator_device,
                torch_dtype=torch.float16,
            )
            model = pipeline(
                'text-generation', 
                model=model,
                tokenizer=tokenizer
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.generator_path,
                quantization_config=bnb_config,
                device_map=self.generator_device,
                torch_dtype=torch.float16,
            )
            model.eval()
        return tokenizer, model, gpt2
    
    def call_retriever(self, question):
        relevant_docs = self.retriever.similarity_search(
            query=question, k=self.num_docs_retrieve
        )
        relevant_docs = [doc.page_content for doc in relevant_docs]

        return relevant_docs
    
    def call_generator(self, prompt):
        outputs = ""
        if self.gpt2:
            prompt = prompt.replace("<|endoftext|>", "") + " Antwort:"
            res = self.generator(prompt, max_new_tokens=50, return_full_text=False,)
            res = res[0]["generated_text"]
        else:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    max_length=4096,
                    truncation=True
                )
                if self.generator_device == "auto" and torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                outputs = self.generator.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=True, 
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            res = self.tokenizer.batch_decode(outputs.detach().cpu().numpy())[0]
            res = res.split(self.assistant_tag)[-1].split(self.answer_end_tag)[0].split("<|")[0]
            if res:
                res = res.strip()
        print("#" * 25)
        print(res)
        print("#" * 25)
        return res

    
    def answer_with_retrieval(self, question):
        relevant_docs = self.call_retriever(question)
        context = "\nExtrahierte Dokumente:\n"
        context += "\n".join(
            [f"Dokument {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
        )
        
        content = "Kontext:\n{context}\n---\nHier ist nun die Frage, die Sie beantworten müssen.\n\nFrage: {question}"
        content = content.format(question=question, context=context)
        
        messages = [
            {
                "role": "system", 
                "content": "Mit den Informationen aus dem Kontext geben Sie eine umfassende Antwort auf die Frage.\nAntworten Sie nur auf die gestellte Frage, Ihre Antwort sollte knapp und relevant zur Frage sein.\nGeben Sie die Nummer des Quelldokuments an, wenn dies relevant ist.\nWenn die Antwort nicht aus dem Kontext abgeleitet werden kann, geben Sie keine Antwort.",
            },
            {
                "role": "user", 
                "content":content
            }
        ]    
        if self.gpt2:
            final_prompt = messages[0]["content"] + "\n" + messages[1]["content"]
        else:
            final_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
            )

        answer = self.call_generator(final_prompt)
        return answer, relevant_docs