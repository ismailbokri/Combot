# Combot  

An intelligent Tunisian law expert and compliance monitoring system powered by LLMs and a curated Tunisian legal dataset.  
It features interactive contract analysis using OCR and RAG-based document understanding.

---

## 📝 Overview  

**Combot** was developed as part of a university project for the *Contract Lifecycle Management* module within the **Management Information Systems** program at **Esprit School of Engineering**.  
It leverages modern technologies to automate the **analysis**, **monitoring**, and **compliance enforcement** of legal documents—reducing human error and operational risks.  

This project aims to make information about Tunisian laws related to contract compliance more accessible and interpretable, with the capability of updating autonomously.

---

## 🚀 Features  

- Uses **Langgraphe** to create a robust, non-sequential workflow  
- Extracts relevant references from Tunisian law to answer contract-related questions  
- Crawls the web to detect updates in Tunisian laws and checks for relevance to the targeted legal codes  
- Automatic extraction of contractual obligations using **OCR** and **NLP**  
- Semantic document search using **Qdrant** and hybrid search techniques  
- Response generation via **LLMs** using **Retrieval-Augmented Generation (RAG)**  
- Lightweight backend implemented in **Flask**

---

## 🧰 Tech Stack  

### ⚙️ Technologies & Tools  

- **Python**  
- **Flask**  
- **Qdrant** (Vector Database)  
- **Langgraphe**, **PyPDF2**, and other RAG pipeline tools  
- **llama**, **DeepSeek**, **BAAI/bge-m3**, **BM25** for model and search components  

### 🧠 Core Techniques  

- **OCR** (PaddleOCR) for scanned document processing  
- **NLP** for understanding contract clauses  
- **LLMs** for chatbot interactions and legal response generation  
- **Hybrid Search** for both dense and keyword-based retrieval  
- **RAG** (Retrieval-Augmented Generation) to generate answers with contextual document references  

---

### 📚 Data  

- **Dataset of contracts** (from Esprit University)  
- **Tunisian legal codes**, including:
  - `loi_defense_contre_pratiques_deloyales_importation`  
  - `loi_relative_Startups`  
  - `loi_relative_commerce_exterieur`  
  - `loi_societes_commerce_international`  
  - `loi_societes_ligne`  
  - `texte_code_societes_commerciales`  

---

## 📁 Directory Structure  :

-` /📁 bm25_states` 
-` /📁 data` 
-` / 📁 evaluation` 
-` / 📁 index` 
-` / 📁 legal_codes` 
-` / 📁 log` 
-` / 📁 static` 
-` / 📁 css` 
-` / 📁 stores` 
-` / 📁 templates` 
-` / 📄 README.md ` 
-` 📄 app.py ` 
-` 📄 combot_functions.py ` 
-` 📄 config.py ` 
-` 📄 contract_functions.py`  
-` 📄 db.py ` 
-` 📄 webCrawlerAgent.py ` 
-` 📄 requirements.txt` 

## ⚙️ Getting Started  

1. Add your **Groq API key** in `config.py`  
2. Install requirements:  
   ```bash
   pip install -r requirements.txt
   
3. Start Qdrant using Docker:
   ```bash
   docker run -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
4. Run the application:  
   ```bash
   python app.py

## 🙏 Acknowledgments
Special thanks to Esprit School of Engineering for the opportunity to build this project and for providing access to the DGX platform, which allowed us to experiment with fine-tuning LLMs using legal texts.

We are grateful to:

- Mr. Morad Zerrai and Mrs. Nardin Hannfi for their mentorship and guidance throughout the project

- Mr. Souhail Oueslati, Group Chief Financial Officer, for offering expert legal and financial advice that shaped our work
