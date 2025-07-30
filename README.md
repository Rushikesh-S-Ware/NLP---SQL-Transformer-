🧠 NLP-SQL Transformer

🔍 Project Summary
NLP-SQL Transformer is a lightweight Gradio-based chatbot system that converts natural language questions into SQL queries over user-uploaded CSV files. Built using a fine-tuned BART Transformer, this project achieves real-time inference with schema-aware serialization and robust query generation.

🚀 Try it live: Hugging Face Space ↗

📌 Features
Upload one or more CSV files

Ask natural language questions like:
"How many rows contain age over 25?"

Get:

Executable SQL query

Corresponding tabular result

Handles fuzzy matching, LIKE queries, and schema ambiguity

Sub-200ms response time with optimized inference

🏗️ Architecture
Fine-tuned BART-large with schema/entity-aware input format

SQLite engine for dynamic CSV execution

Fallbacks for malformed or unmatched queries (LIKE, fuzzy match)

Deployed with Gradio and Hugging Face Spaces

📁 Repository Structure
bash
Copy
Edit
├── app.py                  # Main Gradio app script
├── checkpoint/             # Fine-tuned BART model
├── requirements.txt        # Python dependencies
└── README.md               # This file
📦 Setup Instructions
🧪 Local Testing
bash
Copy
Edit
git clone https://huggingface.co/spaces/Rushikesh-S-Ware/NLP-SQL-Transformer
cd NLP-SQL-Transformer
pip install -r requirements.txt
python app.py
Then open your browser at http://localhost:7860

🧠 Model Details
Base: facebook/bart-large

Trained on: Spider Dataset

Serialization format:

txt
Copy
Edit
[ENT] name [/ENT]
[SCHEMA] t1(name, age, city) | t2(dept, salary) [/SCHEMA]
Question: How many employees named Amy?
📊 Evaluation
Metric	Score
Exact Match (Spider)	45.6%
Execution Accuracy	59.8%
Latency	~95ms

📄 Report and Presentation
📘 Final Report (PDF)

📊 Slide Deck (PPTX)

👥 Team
Rushikesh Ware – rware3@gmu.edu

Saharsh Koli – skoli2@gmu.edu

📜 License
This project is licensed under the MIT License.
