#!/usr/bin/env bash
python evaluate_rag.py --model_to_evaluate "DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental" --assistant_tag "<|im_start|> assistant" &&
python evaluate_rag.py --model_to_evaluate "dbmdz/german-gpt2" --assistant_tag "<|im_start|> assistant" &&
python evaluate_rag.py --model_to_evaluate "LeoLM/leo-mistral-hessianai-7b-chat" --assistant_tag "<|im_start|> assistant" &&
python evaluate_rag.py --model_to_evaluate "DiscoResearch/DiscoLM_German_7b_v1" --assistant_tag "<|im_start|> assistant" &&
python evaluate_rag.py --model_to_evaluate  "LeoLM/leo-hessianai-13b-chat" --assistant_tag "<|im_start|> assistant"

