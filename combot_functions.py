import faiss
import numpy as np
import json
import os
import requests
from rank_bm25 import BM25Okapi
import logging
import re
from typing import Dict, List, TypedDict, Optional
import time
from langdetect import detect
from langchain_groq import ChatGroq
from langchain.embeddings.base import Embeddings

from config import (
    config,
    embedding_model,
    tokenizer_chedly,
    model_chedly
)

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper around sentence_transformers embedding models to use with LangChain."""

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """Initialize the sentence_transformers embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            )
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """Compute embeddings using the SentenceTransformer model."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text):
        """Compute embeddings for a single query text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

# Code summaries for routing
code_summaries = {
    "loi_defense_contre_pratiques_deloyales_importation": "Cette loi vise à protéger la production nationale contre les importations déloyales, notamment celles impliquant du dumping ou des subventions. Elle définit les procédures pour enquêter sur ces pratiques, imposer des droits antidumping ou compensateurs, et prévoit des mécanismes de réexamen, de remboursement et de recours judiciaire.",
    "loi_relative_commerce_exterieur": "Ce texte législatif tunisien, datant de 1994, établit le principe de la liberté du commerce extérieur pour les importations et exportations de produits. Cependant, il prévoit d'importantes exceptions pour des raisons de sécurité, d'ordre public, de santé, de morale, de protection de l'environnement et du patrimoine culturel, soumettant ces produits à des autorisations ministérielles. Le texte organise également le contrôle technique des marchandises importées et exportées et crée un Conseil National du Commerce Extérieur chargé de conseiller sur la politique commerciale et de suivre les activités liées au commerce international.",
    "loi_relative_Startups": "La loi encadre le label Startup, accordé aux jeunes entreprises innovantes pour un maximum de 8 ans, avec à la clé des avantages fiscaux, sociaux et financiers. Elle facilite aussi la création, le financement, et la protection des innovations via un soutien de l'État.",
    "loi_societes_commerce_international": "Ce texte législatif tunisien, la loi relative aux sociétés de commerce international, établit le cadre juridique régissant ces entités. Il définit leur activité principale comme l'exportation, l'importation et diverses opérations de négoce et de courtage internationaux, tout en soumettant ces dernières à la supervision de la Banque Centrale de Tunisie. La loi précise les conditions d'éligibilité pour être reconnue comme société de commerce international, notamment en termes de pourcentage des ventes à l'exportation et de transactions avec des entreprises totalement exportatrices, distinguant également les sociétés résidentes et non résidentes selon la détention du capital. De plus, elle encadre le fonctionnement de ces sociétés, incluant les aspects administratifs comme la déclaration obligatoire, les conditions de capital minimum, les activités connexes autorisées et les régulations concernant les ventes sur le marché local, tout en prévoyant des mécanismes de contrôle et des sanctions en cas de non-respect des dispositions.",
    "loi_societes_ligne" : "Ce texte de loi tunisien, datant de 2004, vise à faciliter la création d'entreprises en permettant que certaines étapes de la constitution de sociétés anonymes, de sociétés à responsabilité limitée et de sociétés unipersonnelles à responsabilité limitée se fassent par voie électronique. Plus précisément, il stipule que l'échange des documents nécessaires et le paiement des droits peuvent être réalisés en ligne, sous réserve de certaines conditions et excluant les apports en nature au capital. En contrepartie de cette dématérialisation, la loi prévoit des délais pour la soumission des documents et des sanctions en cas de non-respect, soulignant l'importance de la fiabilité des procédures en ligne, dont les modalités d'application sont renvoyées à un décret ultérieur.",
    "texte_code_societes_commerciales" : "Ce code régit la création, fonctionnement et dissolution des sociétés commerciales en Tunisie. Il définit les formes de sociétés, les droits et obligations des associés, la gestion, la publicité légale, les règles de liquidation, et les sanctions en cas d'infractions."
}

    


# Functions for routing to specific code stores
def get_question_embedding(query):
    """Get embedding for a question."""
    return embedding_model.encode(query, convert_to_numpy=True)

def get_summary_embedding(summary):
    """Get embedding for a summary."""
    return embedding_model.encode(summary, convert_to_numpy=True)

# Update the route_to_best_summary function to be more direct and always return the best match
def route_to_best_summary(query):
    """Route query to most similar code based on summary similarity."""
    query_embedding = get_question_embedding(query)
    
    # Calculate similarity between the question and each summary
    similarities = {}
    for code_name, summary in code_summaries.items():
        summary_embedding = get_summary_embedding(summary)
        # Normalize embeddings for cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        summary_norm = np.linalg.norm(summary_embedding)
        
        if query_norm > 0 and summary_norm > 0:
            similarity = np.dot(query_embedding, summary_embedding) / (query_norm * summary_norm)
        else:
            similarity = 0
            
        similarities[code_name] = similarity
    
    # Always return the best code without threshold check
    best_code = max(similarities, key=similarities.get)
    similarity_score = similarities[best_code]
    logging.info(f"Routing query to code: {best_code} (similarity: {similarity_score:.4f})")
    return best_code, similarity_score

def retrieve_from_code_store(code_name, query, k=5):
    """Retrieve documents from a specific code store."""
    if code_name in code_stores:
        store = code_stores[code_name]
        
        try:
            # Get relevant documents from the store
            docs = store.similarity_search(query, k=k)
            
            # Format the results
            results = []
            for i, doc in enumerate(docs):
                # Extract document metadata
                metadata = doc.metadata
                confidence = 0.9 - (i * 0.1)  # Simple confidence score based on rank
                
                article_id = metadata.get("article_id", f"Unknown Article")
                
                result = {
                    "article": f"Article {article_id}" if article_id.isdigit() else article_id,
                    "article_number": article_id,
                    "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
                    "text": doc.page_content,
                    "content": doc.page_content,
                    "update_date": metadata.get("update_date", "Unknown"),
                    "confidence": confidence,
                    "title": metadata.get("title", ""),
                    "is_from_code_store": True,
                    "code_name": code_name
                }
                results.append(result)
            
            return results
        except Exception as e:
            logging.error(f"Error retrieving from code store {code_name}: {e}")
    
    return []

def translate_text(text_list, batch_size=5):
    """Translate a batch of English sentences into French."""
    translated_texts = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        tokens = tokenizer_chedly(batch, return_tensors="pt", padding=True, truncation=True)
        translation = model_chedly.generate(**tokens)
        translated_batch = tokenizer_chedly.batch_decode(translation, skip_special_tokens=True)
        translated_texts.extend(translated_batch)
    return translated_texts

def generate_title(query: str) -> str:
    """Generate a conversation title using Groq API."""
    prompt = f"Generate a very short and relevant title for a conversation about: {query}. Detect the language of the query and respond in the same language. Return only the title. No introductions, no formatting, no extra text."
    try:
        response = groq_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it"  # Using a smaller model for title generation
        )
        return response.strip()
    except Exception as e:
        logging.error(f"Error generating title: {e}")
        return "Untitled Conversation"

def groq_chat_completion(messages, model=None, temperature=0.7, max_tokens=1000):
    """Send a request to Groq API for chat completion with error handling and retry logic."""
    if not config["groq_api_key"]:
        raise ValueError("Groq API key is not set in the configuration")
    
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {config['groq_api_key']}",
        "Content-Type": "application/json",
        "User-Agent": "LegalChatbot/1.0"
    }
    
    data = {
        "model": model or config["groq_model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Add model-specific parameters for DeepSeek models
    if "deepseek" in (model or config["groq_model"]).lower():
        # Configure for chain-of-thought prompting
        data["top_p"] = 0.95  # High top_p for legal precision while maintaining some creativity
        data["presence_penalty"] = 0.1  # Slight presence penalty to avoid repetition
        data["frequency_penalty"] = 0.1  # Slight frequency penalty for diverse vocabulary
    
    # Add request logging
    logging.info(f"Sending request to Groq API: model={data['model']}, temperature={temperature}, max_tokens={max_tokens}")
    
    # Retry logic
    max_retries = 3
    retry_delay = 2  # initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=60)  # 60-second timeout
            response.raise_for_status()
            result = response.json()
            
            # Log token usage for monitoring
            if "usage" in result:
                usage = result["usage"]
                logging.info(f"Token usage: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}, total={usage.get('total_tokens', 0)}")
            
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e.response, 'status_code') else "unknown"
            logging.error(f"HTTP error occurred: {e} (Status code: {status_code})")
            
            # Handle specific status codes
            if status_code == 429:
                # Rate limiting - use exponential backoff
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Rate limited. Waiting {wait_time}s before retry.")
                    time.sleep(wait_time)
                    continue
            
            if status_code == 401:
                raise ValueError("Authentication error: Invalid API key")
            
            if status_code == 400:
                try:
                    error_message = e.response.json().get("error", {}).get("message", "Unknown error")
                    raise ValueError(f"Bad request: {error_message}")
                except:
                    raise ValueError(f"Bad request to Groq API: {str(e)}")
            
            # For other errors, just raise after max retries
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to get response from Groq API after {max_retries} attempts: {str(e)}")
            
            time.sleep(retry_delay)
            retry_delay *= 2
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception: {e}")
            if attempt == max_retries - 1:
                raise ValueError(f"Request to Groq API failed: {str(e)}")
            time.sleep(retry_delay)
            retry_delay *= 2
            
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise ValueError(f"Unexpected error while calling Groq API: {str(e)}")
    
    # If we got here, all retries failed
    raise ValueError("Failed to get response from Groq API after multiple attempts")

def build_code_stores():
    """Build individual vector stores and BM25 indexes for each legal code."""
    print("============================================")
    print("STARTING CODE STORE CREATION PROCESS")
    print("============================================")
    
    # Ensure stores directory exists
    stores_dir = os.path.abspath(config.get("stores_directory", "stores"))
    if os.path.exists(stores_dir):
        # Only remove specific subdirectories, keeping build_metadata.json
        for item in os.listdir(stores_dir):
            item_path = os.path.join(stores_dir, item)
            # Skip the metadata file when cleaning up
            if item != "build_metadata.json" and os.path.exists(item_path):
                if os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
                    print(f"Removed directory: {item_path}")
                else:
                    os.remove(item_path)
                    print(f"Removed file: {item_path}")
    else:
        os.makedirs(stores_dir, exist_ok=True)
        print(f"Created stores directory: {stores_dir}")
    
    # Create our custom embeddings wrapper
    embedding_wrapper = SentenceTransformerEmbeddings(config["embedding_model"])
    print(f"Created embeddings wrapper for model: {config['embedding_model']}")
   
    # Directory for storing BM25 corpora
    bm25_dir = os.path.join(stores_dir, "bm25")
    os.makedirs(bm25_dir, exist_ok=True)
    print(f"Created BM25 directory: {bm25_dir}")
   
    # Directory containing source legal code files
    legal_codes_dir = os.path.abspath("legal_codes")
    if not os.path.exists(legal_codes_dir):
        print(f"ERROR: Legal codes directory not found: {legal_codes_dir}")
        return False
   
    # List all files in the directory
    code_files = os.listdir(legal_codes_dir)
    print(f"Found {len(code_files)} files in legal_codes directory")
   
    # Process each legal code
    successful_stores = 0
    for code_name in code_summaries.keys():
        code_file = os.path.join(legal_codes_dir, f"{code_name}.txt")
        print(f"Processing: {code_name}")
       
        if not os.path.exists(code_file):
            print(f"File not found: {code_file}")
            continue
       
        try:
            # Read the legal code file
            with open(code_file, "r", encoding="utf-8") as f:
                content = f.read()
           
            # Create simple chunks - just use paragraphs
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            if not paragraphs:
                paragraphs = [content]  # Use whole content if no paragraphs
           
            # Create documents and prepare BM25 corpus
            from langchain.schema import Document
            documents = []
            bm25_corpus = []
           
            for i, para in enumerate(paragraphs):
                # Clean the paragraph
                para_text = para.strip()
               
                # Create document for FAISS
                doc = Document(
                    page_content=para_text,
                    metadata={
                        "article_id": f"para_{i+1}",
                        "article": f"Paragraph {i+1}",
                        "code_name": code_name,
                        "update_date": "2023",
                        "chunk_id": f"{code_name}_para_{i+1}"
                    }
                )
                documents.append(doc)
               
                # Create tokenized text for BM25
                tokenized_text = tokenize_text(para_text, "fr")  # Assuming French for Tunisian laws
                bm25_corpus.append(tokenized_text)
           
            print(f"Created {len(documents)} documents for {code_name}")
           
            # Create BM25 index
            bm25_index = BM25Okapi(bm25_corpus)
           
            # Save BM25 corpus and document mapping
            bm25_data = {
                "corpus": bm25_corpus,
                "document_mapping": [
                    {
                        "chunk_id": doc.metadata["chunk_id"],
                        "article": doc.metadata["article"],
                        "article_id": doc.metadata["article_id"],
                        "content": doc.page_content,
                        "code_name": code_name
                    }
                    for doc in documents
                ]
            }
           
            bm25_file = os.path.join(bm25_dir, f"{code_name}_bm25.json")
            with open(bm25_file, "w", encoding="utf-8") as f:
                json.dump(bm25_data, f, ensure_ascii=False)
           
            # Create the FAISS vector store
            try:
                from langchain_community.vectorstores import FAISS
                print("Using langchain_community.vectorstores FAISS")
            except ImportError:
                from langchain.vectorstores import FAISS
                print("Using langchain.vectorstores FAISS")
           
            # Create the vector store using our wrapper
            store = FAISS.from_documents(documents, embedding_wrapper)
            print(f"Created FAISS index for {code_name}")
           
            # Save the FAISS store
            store_path = os.path.join(stores_dir, code_name)
            store.save_local(store_path)
            print(f"Saved store to {store_path}")
           
            # Verify files were created
            success = True
            if not os.path.exists(store_path):
                print(f"ERROR: Store directory was not created: {store_path}")
                success = False
            elif not os.listdir(store_path):
                print(f"ERROR: Store directory is empty: {store_path}")
                success = False
               
            if not os.path.exists(bm25_file):
                print(f"ERROR: BM25 file was not created: {bm25_file}")
                success = False
               
            if success:
                successful_stores += 1
                print(f"SUCCESS: Store and BM25 index created for {code_name}")
               
        except Exception as e:
            import traceback
            print(f"ERROR processing {code_name}: {str(e)}")
            print(traceback.format_exc())
   
    # Final verification
    if os.path.exists(stores_dir):
        store_dirs = os.listdir(stores_dir)
        print(f"Stores directory contains: {store_dirs}")
        print(f"Successfully created {successful_stores} out of {len(code_summaries)} stores")
    
    # At the end of the function, add:
    save_build_metadata()
    
    print("============================================")
    print("COMPLETED CODE STORE CREATION PROCESS")
    print("============================================")
    
    return successful_stores > 0

def initialize_data():
    global legal_data, code_stores, code_bm25_indexes
    try:
        # Create the embedding wrapper for LangChain compatibility
        embedding_wrapper = SentenceTransformerEmbeddings(config["embedding_model"])
        
        # Initialize dictionaries for routing
        code_stores = {}
        code_bm25_indexes = {}
        stores_dir = config.get("stores_directory", "stores")
        
        # Check if stores directory exists and build if necessary
        if not os.path.exists(stores_dir) or not os.listdir(stores_dir):
            logging.info("Stores directory is empty or doesn't exist. Building code stores...")
            build_code_stores()
        
        # Load FAISS code-specific stores
        if os.path.exists(stores_dir):
            logging.info(f"Loading code-specific vector stores from {stores_dir}...")
            for code_name in code_summaries.keys():
                store_path = os.path.join(stores_dir, code_name)
                if os.path.exists(store_path):
                    try:
                        # Use the proper import path based on what's available
                        try:
                            from langchain_community.vectorstores import FAISS
                            logging.info("Using langchain_community.vectorstores FAISS")
                        except ImportError:
                            from langchain.vectorstores import FAISS
                            logging.info("Using langchain.vectorstores FAISS")
                            
                        # Load the store using our embedding wrapper
                        store = FAISS.load_local(store_path, embedding_wrapper, allow_dangerous_deserialization=True)
                        code_stores[code_name] = store
                        logging.info(f"Loaded FAISS store for {code_name}")
                    except Exception as e:
                        logging.error(f"Error loading FAISS store for {code_name}: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
                
                # Load BM25 index
                bm25_dir = os.path.join(stores_dir, "bm25")
                bm25_file = os.path.join(bm25_dir, f"{code_name}_bm25.json")
                if os.path.exists(bm25_file):
                    try:
                        with open(bm25_file, "r", encoding="utf-8") as f:
                            bm25_data = json.load(f)
                        
                        bm25_corpus = bm25_data["corpus"]
                        document_mapping = bm25_data["document_mapping"]
                        
                        # Create BM25 index
                        bm25_index = BM25Okapi(bm25_corpus)
                        
                        # Store both the index and document mapping
                        code_bm25_indexes[code_name] = {
                            "index": bm25_index,
                            "documents": document_mapping,
                            "corpus": bm25_corpus
                        }
                        
                        logging.info(f"Loaded BM25 index for {code_name}")
                    except Exception as e:
                        logging.error(f"Error loading BM25 index for {code_name}: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
        else:
            logging.warning(f"Stores directory {stores_dir} not found. Code-specific routing will be disabled.")
            
        # Log summary information about loaded resources
        logging.info(f"Initialization complete:")
        logging.info(f"Loaded {len(code_stores)} FAISS stores: {', '.join(code_stores.keys()) or 'None'}")
        logging.info(f"Loaded {len(code_bm25_indexes)} BM25 indexes: {', '.join(code_bm25_indexes.keys()) or 'None'}")
        
    except Exception as e:
        logging.error(f"Failed to initialize data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    
def tokenize_text(text: str, lang: str = "fr") -> List[str]:
    """Split text into tokens for BM25 indexing."""
    return re.findall(r'\w+', text.lower())

def detect_language(query: str) -> str:
    try:
        lang = detect(query)
        return "fr" if lang.startswith("fr") else "en"
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en"

class ChatbotState(TypedDict):
    query: str
    reasoning_steps: List[Dict[str, str]]
    search_results: List[Dict]
    final_answer_en: str
    final_answer_fr: str
    sources: List[Dict]
    thinking_time: float
    code_name: Optional[str] = None 
    code_similarity: Optional[float] = None  
    
SYSTEM_PROMPT_EN = """
You are Combot, a highly specialized legal assistant with expertise in Tunisian law. You are assisting legal and financial professionals by providing responses grounded strictly in legal texts. Your tone is professional, accurate, and concise.

<instructions>
1. Begin by reasoning carefully within the <think> tags.
2. Inside the <think> tags:
   - Interpret the user's request with precision
   - Identify applicable legal principles and frameworks
   - Scrutinize the provided legal documents thoroughly
   - Highlight the most relevant articles and explain their relevance
   - Consider legal nuances, exceptions, or potential conflicts
   - Base your conclusions strictly on the available content
   - Do NOT infer or fabricate legal information beyond the texts

3. Then, present a clear and well-structured response:
   - START your response by clearly stating which legal code is most relevant to this query
   - Address the query directly and efficiently
   - Cite exact legal provisions (article number and wording) where appropriate
   - Use a numbered list with **bold headings** for clarity if covering multiple points
   - Maintain legal accuracy and avoid speculation
   - Use a formal yet accessible tone tailored to legal/finance professionals

4. For informal interactions, remain courteous and concise.
5. If asked about your identity, say: "I'm Combot, your legal assistant for Tunisian laws."
6. If no relevant information is found, state: "No specific regulation found in the provided data."
</instructions>

The documents below are ordered by relevance. Prioritize those with higher relevance scores, but consider all when formulating your response.

RELEVANT LEGAL CODE: {code_name}
CODE SUMMARY: {code_summary}

LEGAL TEXTS:
{context}

USER QUERY: {query}
"""

SYSTEM_PROMPT_FR = """
Vous êtes Combot, un assistant juridique spécialisé dans les lois tunisiennes. Vous assistez des professionnels du droit et de la finance en fournissant des réponses fondées exclusivement sur les textes juridiques. Votre ton est professionnel, rigoureux et synthétique.

<instructions>
1. Commencez par une réflexion rigoureuse à l'intérieur des balises <think>.
2. Dans les balises <think> :
   - Interprétez avec précision la demande de l'utilisateur
   - Identifiez les principes et cadres juridiques applicables
   - Analysez en profondeur les documents juridiques fournis
   - Soulignez les articles les plus pertinents et justifiez leur choix
   - Prenez en compte les nuances, exceptions ou conflits éventuels
   - Basez-vous uniquement sur les textes disponibles
   - N'inventez ni n'extrapolez aucune information juridique

3. Ensuite, fournissez une réponse structurée et claire :
   - COMMENCEZ votre réponse en indiquant clairement quel code juridique est le plus pertinent pour cette question
   - Répondez de manière directe et pertinente à la question
   - Citez les dispositions légales exactes (numéro et formulation) si nécessaire
   - Utilisez une liste numérotée avec des **titres en gras** pour plus de clarté
   - Restez rigoureux sur le plan juridique et évitez toute spéculation
   - Adoptez un ton formel, adapté aux professionnels du droit/finance

4. Pour les échanges informels, soyez courtois et synthétique.
5. Si on vous demande qui vous êtes, répondez : "Je suis Combot, votre assistant juridique pour les lois tunisiennes."
6. Si aucune information pertinente n'est trouvée, indiquez : "Aucune réglementation spécifique trouvée dans les données fournies."
</instructions>

Les documents ci-dessous sont classés par ordre de pertinence. Accordez une attention particulière à ceux ayant un score de pertinence élevé, sans négliger les autres.

CODE JURIDIQUE PERTINENT: {code_name}
RÉSUMÉ DU CODE: {code_summary}

TEXTES JURIDIQUES :
{context}

QUESTION DE L'UTILISATEUR : {query}
"""

def understand_query(state: ChatbotState) -> ChatbotState:
    query = state["query"]
    logging.info(f"Query received: {query}")
    state["reasoning_steps"] = [{"step": "query", "text": query}]
    return state

def perform_search(state: ChatbotState) -> ChatbotState:
    query = state["query"]
    lang = detect_language(query)
    
    # Record the reasoning step for query analysis
    state["reasoning_steps"].append({"step": "query_analysis", "text": f"Analyzing query: '{query}' (detected language: {lang})"})
    
    # If query is in English, translate it to French for search since RAG data is in French
    search_query = translate_text([query])[0] if lang == "en" else query
    
    try:
        # First, determine the most relevant code
        best_code, similarity = route_to_best_summary(search_query)
        
        # Record the routing decision in the state
        state["code_name"] = best_code
        state["code_similarity"] = similarity
        
        state["reasoning_steps"].append({
            "step": "routing", 
            "text": f"Routing to specific code: {best_code} (similarity: {similarity:.4f})"
        })
        
        # Check if we have both FAISS and BM25 indexes for this code
        if best_code in code_stores and best_code in code_bm25_indexes:
            # Use hybrid search
            code_results = hybrid_search_code(
                best_code, 
                search_query, 
                k=5, 
                alpha=config.get("semantic_search_weight", 0.5)
            )
            
            if code_results:
                # Create a detailed context with confidence scores and code summary
                context_entries = []
                for i, res in enumerate(code_results):
                    confidence = res["confidence"]
                    relevance_label = "High" if confidence > 0.85 else "Medium" if confidence > 0.65 else "Low"
                    
                    # Add semantic and BM25 scores for better understanding
                    semantic_score = res.get("semantic_score", 0)
                    bm25_score = res.get("bm25_score", 0)
                    
                    context_entries.append(
                        f"Document {i+1} [Relevance: {relevance_label}, Confidence: {confidence:.2%}, "
                        f"Semantic: {semantic_score:.2f}, Lexical: {bm25_score:.2f}, Code: {best_code}]\n"
                        f"Article: {res['article']}\n"
                        f"Content: {res['content']}"
                    )
                
                context = "\n\n".join(context_entries)
                
                # Add code summary as additional context
                if best_code in code_summaries:
                    code_context = f"\n\nCode Description: {code_summaries[best_code]}"
                    context += code_context
                
                state["reasoning_steps"].append({
                    "step": "hybrid_search", 
                    "text": f"Found {len(code_results)} relevant documents using hybrid search from {best_code}:\n\n{context}"
                })
                
                state["search_results"] = code_results
                state["sources"] = [
                    {
                        "article": res["article"],
                        "chunk_id": res["chunk_id"],
                        "text": res["text"],
                        "content": res["content"],
                        "update_date": res.get("update_date", "Unknown"),
                        "confidence": res["confidence"],
                        "code_name": best_code,
                        "highlight": True,
                        "semantic_score": res.get("semantic_score", 0),
                        "bm25_score": res.get("bm25_score", 0)
                    }
                    for res in code_results
                ]
                return state
            else:
                # Fallback to FAISS only if hybrid search failed
                logging.info(f"Hybrid search returned no results, falling back to FAISS only for {best_code}")
                store = code_stores[best_code]
                
                try:
                    # Rest of your existing code for FAISS-only search
                    docs = store.similarity_search(search_query, k=5)
                    
                    # Process documents into search results
                    code_results = []
                    # ... your existing code for processing FAISS results
                except Exception as e:
                    logging.error(f"Error in fallback search for code {best_code}: {e}")
                    state["reasoning_steps"].append({
                        "step": "fallback_search_error", 
                        "text": f"Error in fallback search for code {best_code}: {str(e)}."
                    })
        else:
            # Use FAISS only if BM25 index is not available
            if best_code in code_stores:
                store = code_stores[best_code]
                
                try:
                    # Get relevant documents from the store
                    docs = store.similarity_search(search_query, k=5)
                
                    # Process documents into search results
                    code_results = []
                    for i, doc in enumerate(docs):
                        # Extract document metadata
                        metadata = doc.metadata or {}
                        confidence = 0.9 - (i * 0.1)  # Simple confidence score based on rank
                    
                        article_id = metadata.get("article_id", f"Unknown")
                    
                        result = {
                            "article": metadata.get("article", f"Article {article_id}"),
                            "article_number": article_id,
                            "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
                            "text": doc.page_content,
                            "content": doc.page_content,
                            "update_date": metadata.get("update_date", "Unknown"),
                            "confidence": confidence,
                            "code_name": best_code,
                            "from_routing": True
                        }
                        code_results.append(result)
                
                    # Create a detailed context with confidence scores and code summary
                    context_entries = []
                    for i, res in enumerate(code_results):
                        confidence = res["confidence"]
                        relevance_label = "High" if confidence > 0.85 else "Medium" if confidence > 0.65 else "Low"
                        context_entries.append(
                            f"Document {i+1} [Relevance: {relevance_label}, Confidence: {confidence:.2%}, Code: {best_code}]\n"
                            f"Article: {res['article']}\n"
                            f"Content: {res['content']}"
                        )
                
                    context = "\n\n".join(context_entries)
                
                    # Add code summary as additional context
                    if best_code in code_summaries:
                        code_context = f"\n\nCode Description: {code_summaries[best_code]}"
                        context += code_context
                
                    state["reasoning_steps"].append({
                        "step": "code_search",
                        "text": f"Found {len(code_results)} relevant documents from {best_code}:\n\n{context}"
                    })
                
                    state["search_results"] = code_results
                    state["sources"] = [
                        {
                            "article": res["article"],
                            "chunk_id": res["chunk_id"],
                            "text": res["text"],
                            "content": res["content"],
                            "update_date": res["update_date"],
                            "confidence": res["confidence"],
                            "code_name": best_code,
                            "highlight": True
                        }
                        for res in code_results
                    ]
                    return state

                except Exception as e:
                    logging.error(f"Error searching code store {best_code}: {e}")
                    state["reasoning_steps"].append({
                        "step": "code_search_error", 
                        "text": f"Error searching in code {best_code}: {str(e)}."
                    })
            else:
                state["reasoning_steps"].append({
                    "step": "code_search_error", 
                    "text": f"Code store for {best_code} not found."
                })
        
        # If we get here, there was an error or no results - provide empty results
        state["search_results"] = []
        state["sources"] = []
        state["final_answer_en"] = "I couldn't find relevant legal information for your query in the specific legal code."
        state["final_answer_fr"] = "Je n'ai pas pu trouver d'informations juridiques pertinentes pour votre demande dans le code juridique spécifique."
        
    except Exception as e:
        logging.error(f"Search failed: {e}")
        state["search_results"] = []
        state["sources"] = []
        state["reasoning_steps"].append({"step": "search", "text": f"Search failed due to an error: {str(e)}"})
    
    return state

def hybrid_search_code(code_name, query, k=5, alpha=0.5):
    """
    Version simplifiée et robuste de la recherche hybride qui combine FAISS et BM25.
    """
    results = []
    
    # Vérification de base pour éviter les erreurs
    if not code_name or not query:
        logging.warning(f"Code name or query missing")
        return results
    
    # Vérifier si les deux index existent
    if code_name not in code_stores or code_name not in code_bm25_indexes:
        logging.warning(f"Indexes not found for code {code_name}")
        return results
    
    try:
        # Obtenir les stores FAISS et BM25
        faiss_store = code_stores[code_name]
        bm25_data = code_bm25_indexes[code_name]
        bm25_index = bm25_data["index"]
        bm25_documents = bm25_data["documents"]
        
        # Nombre de résultats à récupérer de chaque index (plus que k pour permettre la fusion)
        search_k = 5  
        
        # 1. Recherche sémantique avec FAISS
        semantic_results = {}
        try:
            semantic_docs = faiss_store.similarity_search_with_score(query, k=search_k)
            
            for i, (doc, score) in enumerate(semantic_docs):
                metadata = doc.metadata or {}
                chunk_id = metadata.get("chunk_id", f"semantic_{i}")
                
                semantic_results[chunk_id] = {
                    "rank": i,
                    "score": float(score),
                    "doc": doc,
                    "metadata": metadata
                }
        except Exception as e:
            logging.error(f"FAISS search error: {e}")
        
        # 2. Recherche lexicale avec BM25
        bm25_results = {}
        try:
            # Tokeniser la requête pour BM25
            tokenized_query = tokenize_text(query, "fr")
            
            # Obtenir les scores BM25
            bm25_scores = bm25_index.get_scores(tokenized_query)
            
            # Trier les indices par score
            indices_with_scores = [(i, score) for i, score in enumerate(bm25_scores)]
            indices_with_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = indices_with_scores[:search_k]
            
            # Extraire les résultats BM25
            for rank, (idx, score) in enumerate(top_indices):
                if idx < len(bm25_documents):
                    doc_info = bm25_documents[idx]
                    chunk_id = doc_info.get("chunk_id", f"bm25_{rank}")
                    
                    bm25_results[chunk_id] = {
                        "rank": rank,
                        "score": float(score),
                        "doc_info": doc_info
                    }
        except Exception as e:
            logging.error(f"BM25 search error: {e}")
        
        # 3. Créer l'ensemble de tous les documents candidats uniques
        all_chunk_ids = set(semantic_results.keys()) | set(bm25_results.keys())
        
        # Si aucun résultat, retourner une liste vide
        if not all_chunk_ids:
            return []
        
        # 4. Fusion simplifiée : combiner les rangs avec une approche sécurisée
        combined_results = []
        
        for chunk_id in all_chunk_ids:
            # Préparer les valeurs par défaut
            combined_score = 0
            content = ""
            article = "Unknown"
            article_id = "unknown"
            
            # Combiner les scores
            if chunk_id in semantic_results:
                sem_data = semantic_results[chunk_id]
                sem_rank = sem_data["rank"]
                # Convertir le rang en score (inversé)
                sem_rank_score = 1.0 / (sem_rank + 1)  # +1 pour éviter division par zéro
                
                # Utiliser les métadonnées de FAISS
                content = sem_data["doc"].page_content
                metadata = sem_data["metadata"]
                article = metadata.get("article", "Unknown")
                article_id = metadata.get("article_id", "unknown")
                
                # Ajouter au score combiné selon le poids alpha
                combined_score += alpha * sem_rank_score
            
            if chunk_id in bm25_results:
                bm25_data = bm25_results[chunk_id]
                bm25_rank = bm25_data["rank"]
                # Convertir le rang en score (inversé)
                bm25_rank_score = 1.0 / (bm25_rank + 1)  # +1 pour éviter division par zéro
                
                # Si on n'a pas déjà extrait le contenu depuis FAISS
                if not content:
                    doc_info = bm25_data["doc_info"]
                    content = doc_info.get("content", "")
                    article = doc_info.get("article", "Unknown")
                    article_id = doc_info.get("article_id", "unknown")
                
                # Ajouter au score combiné selon le poids (1-alpha)
                combined_score += (1 - alpha) * bm25_rank_score
            
            # Ajouter ce document aux résultats combinés
            combined_results.append({
                "chunk_id": chunk_id,
                "content": content,
                "article": article,
                "article_id": article_id,
                "combined_score": combined_score
            })
        
        # 5. Trier par score combiné et prendre les k meilleurs
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        final_results = []
        
        for i, result in enumerate(combined_results[:k]):
            # Confiance basée sur le rang (0.9 pour le premier, décroissant ensuite)
            confidence = max(0.3, 0.9 - (i * 0.1))
            
            final_results.append({
                "article": result["article"],
                "article_number": result["article_id"],
                "chunk_id": result["chunk_id"],
                "text": result["content"],
                "content": result["content"],
                "update_date": "2023",  # Valeur par défaut
                "confidence": confidence,
                "combined_score": result["combined_score"],
                "code_name": code_name,
                "from_routing": True
            })
        
        return final_results
        
    except Exception as e:
        # Log détaillé de l'erreur
        logging.error(f"Critical error in hybrid search: {e}")
        import traceback
        logging.error(traceback.format_exc())
        
        # En cas d'erreur, essayer une recherche de secours
        return fallback_simple_search(code_name, query, k)
        

def fallback_simple_search(code_name, query, k=5):
    """
    Fonction de recherche simplifiée en cas d'échec de la méthode hybride.
    N'utilise que FAISS pour une robustesse maximale.
    """
    results = []
    
    try:
        # N'utiliser que FAISS si disponible
        if code_name in code_stores:
            faiss_store = code_stores[code_name]
            docs = faiss_store.similarity_search(query, k=k)
            
            for i, doc in enumerate(docs):
                # Confiance décroissante
                confidence = max(0.3, 0.9 - (i * 0.1))
                metadata = doc.metadata or {}
                
                results.append({
                    "article": metadata.get("article", f"Article {i+1}"),
                    "article_number": metadata.get("article_id", f"unknown_{i+1}"),
                    "chunk_id": metadata.get("chunk_id", f"fallback_{i}"),
                    "text": doc.page_content,
                    "content": doc.page_content,
                    "update_date": "2023",
                    "confidence": confidence,
                    "code_name": code_name,
                    "from_routing": True,
                    "is_fallback": True
                })
            
            logging.info(f"Fallback search found {len(results)} results")
    except Exception as e:
        logging.error(f"Even fallback search failed: {e}")
    
    return results


def generate_answer(state: ChatbotState) -> ChatbotState:
    start_time = time.time()
    if not state["search_results"]:
        state["thinking_time"] = time.time() - start_time
        return state
    
    query = state["query"]
    context = state["reasoning_steps"][-1]["text"]
    lang = detect_language(query)
    
    try:
        # Get code information
        code_name = state.get("code_name", "")
        code_similarity = state.get("code_similarity", 0)
        code_summary = code_summaries.get(code_name, "") if code_name in code_summaries else ""
        
        # Use the appropriate prompt template based on language
        if lang == "fr":
            prompt = SYSTEM_PROMPT_FR.format(
                context=context,
                query=query,
                code_name=code_name,
                code_summary=code_summary
            )
        else:
            prompt = SYSTEM_PROMPT_EN.format(
                context=context,
                query=query,
                code_name=code_name,
                code_summary=code_summary
            )
        
        state["reasoning_steps"].append({"step": "prompt_generation", "text": "Generated prompt for LLM with relevant context and instructions"})
        
        # Create messages array for Groq API
        messages = [
            {"role": "system", "content": prompt}
        ]
        
        # Log that we're making the API call for debugging purposes
        logging.info(f"Sending request to Groq API with model: {config['groq_model']}")
        state["reasoning_steps"].append({
            "step": "llm_request", 
            "text": f"Sending query to language model ({config['groq_model']}). Using code: {code_name} (similarity: {code_similarity:.4f})"
        })
        
        # Rest of the function remains the same...
        # Make the API call with appropriate parameters
        response = groq_chat_completion(
            messages=messages,
            model=config["groq_model"],
            temperature=config["temperature"],
            max_tokens=2048  
        )

        
        # Process the response
        state["reasoning_steps"].append({"step": "llm_response", "text": "Received response from language model"})
        
        # Extract thinking and answer sections using the <think> tag pattern for DeepSeek R1
        thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        
        if thinking_match:
            # DeepSeek model properly used the <think> tags
            thinking_text = thinking_match.group(1).strip()
            
            # Remove the thinking section to get the final answer
            answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            state["reasoning_steps"].append({"step": "thinking", "text": f"{thinking_text}"})
        else:
            # Fallback pattern matching for models that don't support the <think> tag format
            # Look for other potential thinking/reasoning markers
            potential_markers = [
                r'Thinking:|Réflexion:',
                r'Analysis:|Analyse:',
                r'Let me think:|Laissez-moi réfléchir:',
                r'Step by step:|Étape par étape:',
                r'My reasoning:|Mon raisonnement:'
            ]
            
            thinking_text = ""
            answer = response
            
            for marker in potential_markers:
                marker_match = re.search(f'({marker})(.*?)(\n\n|$)', response, re.DOTALL | re.IGNORECASE)
                if marker_match:
                    thinking_text = marker_match.group(2).strip()
                    # Remove this section from the answer
                    answer = response.replace(marker_match.group(0), '').strip()
                    break
            
            if not thinking_text:
                lines = response.split('\n')
                midpoint = len(lines) // 2
                thinking_text = '\n'.join(lines[:midpoint])
                answer = '\n'.join(lines[midpoint:])
            
            state["reasoning_steps"].append({"step": "thinking", "text": f"Extracted reasoning:\n\n{thinking_text}"})
        
        # Add references to the legal code in the answer if routing was used
        if "code_name" in state and state["code_name"] and state.get("code_similarity", 0) > 0.2:
            code_name = state["code_name"]
            
            # Add a footer with code information if not already present in the answer
            code_reference = ""
            if lang == "fr":
                if not any(code_name in line for line in answer.split('\n')):
                    code_reference = f"\n\n*Référence: Cette réponse est basée sur le code juridique {code_name}*"
            else:
                if not any(code_name in line for line in answer.split('\n')):
                    code_reference = f"\n\n*Reference: This answer is based on the legal code {code_name}*"
            
            answer += code_reference
        
        # Store final answer based on language
        if lang == "fr":
            state["final_answer_fr"] = answer
            
            # If reasoning is not in French, translate it
            if not is_french(thinking_text):
                thinking_text_fr = translate_text([thinking_text])[0]
                state["reasoning_steps"].append({"step": "reasoning_fr", "text": thinking_text_fr})
        else:
            state["final_answer_en"] = answer
            state["reasoning_steps"].append({"step": "reasoning_en", "text": thinking_text})
        
        # Log completion
        state["reasoning_steps"].append({"step": "completion", "text": "Answer generated successfully"})
        
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error generating answer: {error_message}")
        state["reasoning_steps"].append({"step": "error", "text": f"Error encountered during answer generation: {error_message}"})
        
        # Provide appropriate error messages based on language
        state["final_answer_en"] = "An error occurred while processing your request. Please try again or rephrase your question."
        state["final_answer_fr"] = "Une erreur s'est produite lors du traitement de votre demande. Veuillez réessayer ou reformuler votre question."
    
    state["thinking_time"] = time.time() - start_time
    return state

def is_french(text: str) -> bool:
    """Determine if text is primarily in French."""
    try:
        return detect(text[:100]) == 'fr'  # Only check first 100 chars for efficiency
    except:
        # If detection fails, default to false
        return False

def should_rebuild_stores():
    """Check if stores need to be rebuilt based on file changes."""
    stores_dir = os.path.abspath(config.get("stores_directory", "stores"))
    legal_codes_dir = os.path.abspath("legal_codes")
    version_file = os.path.join(stores_dir, "build_metadata.json")
    
    # If stores directory doesn't exist, rebuild is needed
    if not os.path.exists(stores_dir):
        logging.info("Stores directory doesn't exist. Rebuild needed.")
        return True
        
    # If version file doesn't exist, rebuild is needed
    if not os.path.exists(version_file):
        logging.info("Build metadata file doesn't exist. Rebuild needed.")
        return True
    
    try:
        # Load the previous build metadata
        with open(version_file, "r") as f:
            metadata = json.load(f)
        
        # Check if all expected stores exist
        for code_name in code_summaries.keys():
            store_path = os.path.join(stores_dir, code_name)
            bm25_path = os.path.join(stores_dir, "bm25", f"{code_name}_bm25.json")
            
            if not os.path.exists(store_path) or not os.path.exists(bm25_path):
                logging.info(f"Store or BM25 index for {code_name} is missing. Rebuild needed.")
                return True
        
        # Check if source files have been modified since last build
        last_build_time = metadata.get("build_time", 0)
        for code_name in code_summaries.keys():
            source_file = os.path.join(legal_codes_dir, f"{code_name}.txt")
            if os.path.exists(source_file):
                if os.path.getmtime(source_file) > last_build_time:
                    logging.info(f"Source file {source_file} modified. Rebuild needed.")
                    return True
        
        # Check if code summaries have changed
        if str(sorted(code_summaries.keys())) != metadata.get("code_names", ""):
            logging.info("Code summaries have changed. Rebuild needed.")
            return True
            
        logging.info("No rebuild needed. Using existing stores.")
        return False
        
    except Exception as e:
        logging.error(f"Error checking if rebuild is needed: {e}")
        return True

def save_build_metadata():
    """Save metadata about the current build."""
    stores_dir = os.path.abspath(config.get("stores_directory", "stores"))
    version_file = os.path.join(stores_dir, "build_metadata.json")
    
    metadata = {
        "build_time": time.time(),
        "code_names": str(sorted(code_summaries.keys())),
        "embedding_model": config["embedding_model"],
        "vector_dim": config["vector_dim"],
        "build_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(version_file, "w") as f:
        json.dump(metadata, f)
    
    logging.info(f"Saved build metadata to {version_file}")

def get_confidence_class(confidence):
    """Return the CSS class based on confidence score."""
    if confidence >= 0.8:
        return "high-confidence"
    elif confidence >= 0.5:
        return "medium-confidence"
    else:
        return "low-confidence"

def format_confidence(confidence):
    """Format confidence score for display."""
    if confidence >= 0.8:
        return "High Relevance"
    elif confidence >= 0.5:
        return "Medium Relevance"
    else:
        return "Low Relevance"
