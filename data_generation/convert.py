import os
import pymupdf4llm
import pandas as pd

for doc in os.listdir("docs"):
    md_text = pymupdf4llm.to_markdown(os.path.join("docs", doc))
    with open(os.path.join("markdown", doc.replace(".pdf", ".md")), "w", encoding="utf-8") as f:
        f.write(md_text)

markdown_path = "markdown"
markdown_file_names = os.listdir(markdown_path)

markdown_files = []
for markdown_file_name in markdown_file_names:
    markdown_file_path = os.path.join(markdown_path, markdown_file_name)
    with open(markdown_file_path, 'r', encoding="utf-8") as m_file:
        markdown_files.append([markdown_file_name, m_file.read()])
markdown_files = pd.DataFrame(markdown_files, columns=["source", "text"])
markdown_files.to_csv("documents.csv", index=False)