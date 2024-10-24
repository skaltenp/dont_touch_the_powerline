#!/usr/bin/env bash
python generate_rag.py --generator_path "DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental" &&
python generate_rag.py --generator_path "dbmdz/german-gpt2" --num_docs_retrieve 1 &&
python generate_rag.py --generator_path "LeoLM/leo-mistral-hessianai-7b-chat" --assistant_tag "<|im_start|> assistant" &&
python generate_rag.py --generator_path "DiscoResearch/DiscoLM_German_7b_v1" --assistant_tag "<|im_start|> assistant" &&
python generate_rag.py --generator_path  "LeoLM/leo-hessianai-13b-chat" --assistant_tag "<|im_start|> assistant"

