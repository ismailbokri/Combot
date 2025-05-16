import os
import json
import yaml
import torch
import logging
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import MarianTokenizer, MarianMTModel
from langchain_groq import ChatGroq

# Charger les variables d'environnement
load_dotenv()
api_key = "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"

# Chemins et paramètres
output_dir = os.path.join('.', 'uploads')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
DIMENSION = 1024
TARGET_CHUNK_SIZE = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DOCS_METADATA_PATH = "data/documents_metadata.json"
CONFIG_FILE = "config.yaml"
BATCH_SIZE = 32
MAX_CHUNK_SIZE = 600
OVERLAP_SIZE = 50
DENSE_MODEL_NAME = "BAAI/bge-m3"

# Initialisation des objets
model_llama = ChatGroq(model_name="llama3-70b-8192", groq_api_key=api_key, temperature=0.7)
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.7)
ocr = PaddleOCR(use_angle_cls=True, lang="fr")
dense_model = HuggingFaceEmbeddings(model_name=DENSE_MODEL_NAME)
client = QdrantClient("http://localhost:6333", timeout=60)

# Logging
logging.basicConfig(
    level=logging.INFO,
    filename="legal_chatbot.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename="legal_chatbot.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

code_bm25_indexes = {} 

# Load configuration
DEFAULT_CONFIG = {
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "vector_dim": 384,
    "groq_api_key": "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",
    "groq_model": "deepseek-r1-distill-llama-70b",  
    "embedding_batch_size": 8,
    "search_top_k": 5,
    "semantic_search_weight": 0.5,
    "lexical_search_weight": 0.5,
    "legal_terms_bonus_weight": 0.2,
    "temperature": 0.1,
    "stores_directory": "stores"
}


config = DEFAULT_CONFIG

# Chargement des métadonnées des documents
with open(DOCS_METADATA_PATH, 'r', encoding='utf-8') as f:
    documents_metadata = json.load(f)

# Modèle de traduction
model_name_chedly = "Helsinki-NLP/opus-mt-en-fr"
tokenizer_chedly = MarianTokenizer.from_pretrained(model_name_chedly)
model_chedly = MarianMTModel.from_pretrained(model_name_chedly)

# Modèle d'embedding
embedding_model = SentenceTransformer(config["embedding_model"])
