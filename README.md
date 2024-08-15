# Don't Touch the Power line

## Results of Open Source Evaluation using ABB data
| Models             |   Normalized Score |   Actual Score |
|:-------------------|-------------------:|---------------:|
| GPT2               |              0.421 |          2.684 |
| LeoLM-Mistral-7b   |              0.508 |          3.032 |
| LeoLM-LaMA2-13b    |              0.516 |          3.063 |
| LlaMA3-DiscoLM-8b  |              0.53  |          3.121 |
| DiscoLM-Mistral-7b |              0.542 |          3.168 |

![Boxplot Results](images/results_box_plot.png)

![Histogram Results](images/results_hist_plot.png)

## 1 Installation

Install using conda.

```
conda env create -f dtpl.yml
conda activate dtpl
```

## 2 Generate HuggingFace Token

Generate a your huggingface token (can be found [here](https://huggingface.co/docs/hub/en/security-tokens)).

## 3 Generate your datasets

Download all files from "data_generation/links.csv" to "data_generation/docs" (you may automate this process :blush:) then run
```
cd data_generation
python convert.py
```

## 4 Generate Questions and Answers
Run
```
cd ../
python 01_generate_qa.py
```
At the beginning you will be prompted to enter the huggingface access token.

## 5 Validate Questions and Answers
Run
```
python 02_validate_qa.py
```

## 6 Generate the RAG Answers of the 5 Models
Run
```
sh 03_generate_rag.sh
```

## 7 Generate the RAG Answers of the 5 Models
Run
```
sh 04_evaluate_rag.sh
```

## 8 Calculate Result Measures
Run all cells in _05_calculate_results.ipynb_.

## Other Information
If you have problems with running the scripts contact [sascha.kaltenpoth@uni-paderborn.de](mailto:sascha.kaltenpoth@uni-paderborn.de).
If you want to only get the measurements just start with the step 8.