# RAG Document Chatbot (Streamlit)

Files:
- app.py               -> main Streamlit app
- requirements.txt     -> dependencies
- make_sample_pdf.py   -> (optional) regenerates sample_document.pdf
- sample_document.pdf  -> sample document (topic: Machine Learning Basics)

## Run locally
1. python -m venv venv
2. venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
3. pip install -r requirements.txt
4. streamlit run app.py
5. Open http://localhost:8501, upload sample_document.pdf, ask questions.

First run downloads the models (~1-3 GB), so it needs internet once.
