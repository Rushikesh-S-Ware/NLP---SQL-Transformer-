# 🧠 NLP-SQL Transformer

A Gradio-powered chatbot that transforms natural language questions into SQL queries, enabling users to explore their uploaded CSV data interactively.

## 📄 Project Summary

**NLP-SQL Transformer** is an intelligent question-answering system that allows users to upload one or more CSV files and ask data-related questions in plain English (e.g., _"How many users are over 25?"_). Powered by a fine-tuned BART Transformer model, it dynamically generates SQL queries and executes them on the uploaded files, returning both the query and the result.

➡️ **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Rushikesh-S-Ware/NLP-SQL-Transformer)**

---

## 🚀 Features

- 📂 Upload one or more CSV files
- 🧠 Ask natural language questions
- 🧾 Generates executable SQL queries
- 📊 Returns tabular results
- 🧩 Handles fuzzy matching, LIKE queries, and ambiguous schema
- ⚡ Optimized inference with sub-200ms response times

---

## 🗂️ Project Structure

| File/Folder                    | Description                                                      |
|-------------------------------|------------------------------------------------------------------|
| `app.py`                      | Main application logic for Gradio interface and model inference |
| `Model_Training.ipynb` | Model training & inference notebook                              |
| `Demo_Notebook.ipynb`  | Demonstration and usage examples notebook                        |
| `Project_Overview.txt`         | Project notes and experiment logs                                |
| `Model_Checkpoint.zip`                 | Fine-tuned BART model directory                                 |
| `requirements.txt`            | Python dependencies for deployment                              |

---
## 🧠 System Architecture




## 🧠 How It Works

1. **Upload CSV Files** – These are loaded into an in-memory SQLite database.
2. **Ask a Question** – The system constructs a schema-aware prompt.
3. **Generate SQL** – The BART model generates a SQL query based on the input.
4. **Post-process Query** – Applies regex corrections and fuzzy logic if needed.
5. **Display Results** – Executes the query and returns the results in a table.

---

## 📦 Installation (Local)

```bash
git clone https://huggingface.co/spaces/Rushikesh-S-Ware/NLP-SQL-Transformer
cd NLP-SQL-Transformer
pip install -r requirements.txt
python app.py
```

---

## 📜 License

MIT License
