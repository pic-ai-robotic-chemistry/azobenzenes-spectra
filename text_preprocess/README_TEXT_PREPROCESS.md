We used these scripts to process the research articles we downloaded in PDF format, into cleaned Markdown texts that are ready to use for GraphRAG knowledge graph creation.
**How to use**
Required package "docling", installed via
```
pip install docling
```
Download the scripts to an empty directory, and place your downloaded PDF files in a subfolder named "pdf" under that directory.
First run "1_convert_to_markdown.py" to convert PDF files to Markdown, then run "2_markdown_clean_file.py" on each file for clutter removal, then run "3_additional_cleaning.py md graphrag_in" to remove unwanted parts in GraphRAG indexing and save the inputs for indexing.
Paste these commands to run everything in a row:
```
python 1_convert_to_markdown.py
for file in md/*; do python 2_markdown_clean_file.py ${file} ; done
python 3_additional_cleaning.py md graphrag_in
```