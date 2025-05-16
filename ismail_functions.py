
import numpy as np
import json
import os
from rank_bm25 import BM25Okapi
import re
from typing import Dict, List, TypedDict, Annotated
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import operator
import PyPDF2
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv


from config import (
    model_llama,
    llm,
    ocr,
    output_dir,
    BATCH_SIZE,
    MAX_CHUNK_SIZE,
    dense_model,
    client
)



def sanitize_index_name(filename: str) -> str:
    """Génère un nom de collection valide à partir du nom de fichier"""
    base = os.path.splitext(filename)[0]
    cleaned = re.sub(r'[^a-zA-Z0-9-]', '-', base).lower()
    return f"{cleaned}"[:45]

def _chunk_file(file_path: str) -> List[dict]:
    """Découpage amélioré avec gestion des phrases et chevauchement contextuel"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = re.sub(r'\s+', ' ', text).strip()
    
    sentences = []
    start = 0
    for match in re.finditer(r'[.!?…](?:\s+|$)', text):
        end = match.end()
        sentences.append({
            "text": text[start:end].strip(),
            "start": start,
            "end": end
        })
        start = end
    
    if start < len(text):
        sentences.append({
            "text": text[start:].strip(),
            "start": start,
            "end": len(text)
        })

    chunks = []
    chunk_id = 0
    window_size = 3
    overlap = 1

    for i in range(0, len(sentences), window_size - overlap):
        chunk_sentences = sentences[i:i+window_size]
        chunk_text = ' '.join(s['text'] for s in chunk_sentences)
        
        if len(chunk_text) > MAX_CHUNK_SIZE:
            chunk_text = chunk_text[:MAX_CHUNK_SIZE]
            last_space = chunk_text.rfind(' ', 0, MAX_CHUNK_SIZE-50)
            if last_space != -1:
                chunk_text = chunk_text[:last_space].strip()
        
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "source": os.path.basename(file_path),
            "start_pos": chunk_sentences[0]['start'],
            "end_pos": chunk_sentences[-1]['end']
        })
        chunk_id += 1

    merged_chunks = []
    prev_chunk = None
    for chunk in chunks:
        if prev_chunk and len(prev_chunk['text']) < MAX_CHUNK_SIZE//2:
            prev_chunk['text'] += ' ' + chunk['text']
            prev_chunk['end_pos'] = chunk['end_pos']
            if len(prev_chunk['text']) > MAX_CHUNK_SIZE:
                prev_chunk['text'] = prev_chunk['text'][:MAX_CHUNK_SIZE]
        else:
            if prev_chunk:
                merged_chunks.append(prev_chunk)
            prev_chunk = chunk
    if prev_chunk:
        merged_chunks.append(prev_chunk)

    return merged_chunks

# Modèle BM25 personnalisé ----------------------------------------------------
class BM25Embedder:
    def __init__(self, corpus):
        self.tokenized_corpus = [self.tokenize(doc["text"]) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.vocab = self._create_vocab()
    
    @staticmethod
    def tokenize(text):
        return text.lower().split()
    
    def _create_vocab(self):
        all_tokens = set()
        for tokens in self.tokenized_corpus:
            all_tokens.update(tokens)
        return {token: idx for idx, token in enumerate(sorted(all_tokens))}
    
    def embed(self, text):
        tokenized = self.tokenize(text)
        scores = self.bm25.get_scores(tokenized)
        
        score_dict = {}
        for token, score in zip(tokenized, scores):
            if token in self.vocab:
                idx = self.vocab[token]
                score_dict[idx] = score_dict.get(idx, 0) + float(score)
        
        indices = sorted(score_dict.keys())
        values = [score_dict[idx] for idx in indices]
        
        return models.SparseVector(indices=indices, values=values)

# Gestion des index ----------------------------------------------------------
def check_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        return client.collection_exists(collection_name)
    except Exception as e:
        print(f"Erreur de vérification de la collection: {e}")
        return False

def load_or_create_chunks(collection_name: str, data_file: str) -> List[dict]:

    chunk_file = os.path.join("bm25_states", f"{collection_name}_chunks.pkl")

    if os.path.exists(chunk_file):
        print(f"Chargement des chunks existants depuis {chunk_file}")
        with open(chunk_file, "rb") as f:
            return pickle.load(f)
    else:
        print("Découpage du fichier...")
        chunks = _chunk_file(data_file)
        with open(chunk_file, "wb") as f:
            pickle.dump(chunks, f)
        return chunks

def load_or_create_bm25(collection_name: str, chunks: List[dict]) -> BM25Embedder:

    model_file = os.path.join("bm25_states", f"{collection_name}_bm25.pkl")
    if os.path.exists(model_file):
        print(f"Chargement du modèle BM25 existant depuis {model_file}")
        with open(model_file, "rb") as f:
            return pickle.load(f)
    else:
        print("Création du nouveau modèle BM25...")
        bm25_embedder = BM25Embedder(chunks)
        with open(model_file, "wb") as f:
            pickle.dump(bm25_embedder, f)
        return bm25_embedder

def index_data(client: QdrantClient, chunks: List[dict], collection_name: str, bm25_embedder: BM25Embedder, dense_model):
    if check_collection_exists(client, collection_name):
        print(f"La collection {collection_name} existe déjà, skip de l'indexation")
        return
    
    embedding_dimension = len(dense_model.embed_query("test"))
    
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=embedding_dimension,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                    full_scan_threshold=20000
                )
            )
        }
    )
    
    points = []
    for batch in tqdm.tqdm([chunks[i:i+BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)], 
                          desc="Indexation"):
        batch_texts = [doc["text"] for doc in batch]
        dense_embeddings = dense_model.embed_documents(batch_texts)
        sparse_embeddings = [bm25_embedder.embed(doc["text"]) for doc in batch]
        
        for doc, dense_vec, sparse_vec in zip(batch, dense_embeddings, sparse_embeddings):
            points.append(PointStruct(
                id=doc["id"],
                vector={
                    "dense": dense_vec,
                    "bm25": sparse_vec
                },
                payload=doc
            ))
        
        if len(points) >= 100:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            points = []
    
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )

# Recherche hybride corrigée --------------------------------------------------
def hybrid_search(client: QdrantClient, query: str, collection_name: str, bm25_embedder: BM25Embedder, 
                 dense_model: HuggingFaceEmbeddings, alpha: float = 0.5, top_k: int = 5):
    """Version corrigée avec l'API actuelle de Qdrant"""
    # Génération des embeddings
    dense_embedding = dense_model.embed_query(query)
    sparse_embedding = bm25_embedder.embed(query)

    # Recherche dense
    dense_results = client.search(
        collection_name=collection_name,
        query_vector=models.NamedVector(
            name="dense",
            vector=dense_embedding
        ),
        limit=top_k * 2,
        with_payload=True,
        score_threshold=0.0
    )

    # Recherche sparse
    sparse_results = client.search(
        collection_name=collection_name,
        query_vector=models.NamedSparseVector(
            name="bm25",
            vector=models.SparseVector(
                indices=sparse_embedding.indices,
                values=sparse_embedding.values
            )
        ),
        limit=top_k * 2,
        with_payload=True,
        score_threshold=0.0
    )

    # Normalisation des scores
    def normalize_scores(results):
        scores = [r.score for r in results]
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        return [(s - min_score) / (max_score - min_score + 1e-6) for s in scores]
    
    dense_scores = normalize_scores(dense_results)
    sparse_scores = normalize_scores(sparse_results)

    # Fusion des résultats
    combined = {}
    for result, score in zip(dense_results, dense_scores):
        combined[result.id] = {
            "dense_score": score,
            "sparse_score": 0.0,
            "payload": result.payload,
            "score": result.score
        }

    for result, score in zip(sparse_results, sparse_scores):
        if result.id in combined:
            combined[result.id]["sparse_score"] = score
        else:
            combined[result.id] = {
                "dense_score": 0.0,
                "sparse_score": score,
                "payload": result.payload,
                "score": result.score
            }

    # Calcul du score combiné
    for item in combined.values():
        item["combined_score"] = alpha * item["dense_score"] + (1 - alpha) * item["sparse_score"]

    sorted_results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)[:top_k]
    
    return sorted_results







##########################################################################################################################

### Set up our OCR class, which contains the constructor and the processing function that handles the PDF and converts it into text .
class PDFProcessor:
    def __init__(self, ocr_instance):
        self.ocr = ocr_instance
    
    def process_pdf(self, pdf_file: str, page_number: int) -> str:
        if not os.path.isfile(pdf_file):
            return f"Le fichier {pdf_file} n'existe pas."
        
        try:
            images = convert_from_path(pdf_file)
            if page_number < 1 or page_number > len(images):
                return f"Numéro de page {page_number} invalide"
            
            image_np = np.array(images[page_number - 1])
            result = self.ocr.ocr(image_np, cls=True)
            return "\n".join([line[1][0] for line in result[0]]) if result and result[0] else ""
        
        except Exception as e:
            return f"Erreur lors du traitement : {str(e)}"

ocr_tool = PDFProcessor(ocr)



##############################################################################
######################## Define our functions ################################
##############################################################################

### State of the graph  
class State(TypedDict):
    messages: Annotated[list, operator.add]
    page_number: int
    total_pages: int
    pdf_file: str

### Function to count the number of pages in a document.
 
def count_total_pages(pdf_file: str) -> int:
    with open(pdf_file, 'rb') as f:
        return len(PyPDF2.PdfReader(f).pages)

### Function to get the file name . 
def get_pdf_file_name(filename: str) -> str:  
    return filename[:-4] if filename.endswith('.pdf') else filename

### Function to extract text from pages and update the state.
def extract_page(state: State):
    try:
        result = ocr_tool.process_pdf(
            state["pdf_file"], 
            state["page_number"]
        )
        
        state["messages"].append(SystemMessage(content=result))
        return {"page_number": state["page_number"] + 1}

    except Exception as e:
        return {"error": str(e)}

###Function to determine whether processing should continue.
def should_continue(state: State) -> str:
    return "continue" if state["page_number"] <= state["total_pages"] else "end"    


### Fonction de recherche sur les contrats
def fulltext_search(query, data, exact=False):
    search_fields = {
        'fichier': data.get('source_fichier', ''),
        'titre': data.get('titre', ''),
        'contexte': data.get('contexte', ''),
        'entites': ' '.join(data.get('entites_cles', [])),
        'date': data.get('date', ''),
        'type': data.get('type_document', '')
    }
    
    for field, value in search_fields.items():
        value = str(value).lower()
        if exact:
            if query == value:
                return True
        else:
            if query in value:
                return True
    return False


def merge_json_files():

    input_folder = os.path.join('.', 'data')
    output_folder = os.path.join('.', 'uploads')
    output_filename = 'documents_metadata.json'



    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    merged_data = []

    for json_file in json_files:
        file_path = os.path.join(input_folder, json_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                merged_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Erreur de lecture dans le fichier {json_file}: {e}")

    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, indent=4, ensure_ascii=False)
    



def load_contracts():

    folder = os.path.join('.', 'data')
    filename = 'documents_metadata.json'

    file = os.path.join(folder, filename)

    with open(file, 'r', encoding='utf-8') as f:
        contracts = json.load(f)
    return contracts


### Function run_llm_ocr: a function that runs the LLM to correct the extracted text and convert it into the desired format.
def run_llm_ocr(state: State):
    try:
        last_message = state["messages"][-1]
        
        messages = [
            SystemMessage(content="""[FR/EN] 
1. Correction orthographe/grammaire SEULEMENT
2. CONSERVER la structure originale :
   - Paragraphes complets
   - Listes numérotées
   - Titres formatés (Article 1, Section 2.3)
3. SUPPRIMER ABSOLUMENT :
   - Timbres/signatures/codes administratifs
   - Texte isolé (.s.i.a., ...nak..., QuaA.sant)
   - Lignes de séparation (----, ____)
   - Mentions techniques (Visé pour timbre...)
4. Ne JAMAIS ajouter :
   - Commentaires
   - Phrases d'introduction/transition
   - Éléments de formatage
5. Sortie = Texte original corrigé, structuré et filtré
6. Ne pas ajouter du texte comme "TEXTE CORRIGÉ ET FILTRÉ:" au debut de votre reponse     
7.Conserver les parties relatives au dates .
8.Faire le retour a la ligne pour chaque partie .                                
                                      """),
            HumanMessage(content=f"TEXTE BRUT:\n{last_message.content}\n\nTEXTE CORRIGÉ ET FILTRÉ:")
        ]
        
        response = model_llama.invoke(messages)
        
        output_path = os.path.join(
            output_dir, 
            f"{get_pdf_file_name(state['pdf_file'])}.txt"
        )
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(response.content.strip() + "\n\n")

        state["messages"].append(AIMessage(content=response.content))
        return {"next": "extract_page"}
    
    except Exception as e:
        return {"error": str(e)}

###Function run_llm_describer: a function that runs the LLM to generate a description of the contract and extract key information such as title, date, etc.
def run_llm_describer(state: State):
    """Agent de description de document qui génère des métadonnées structurées"""
    try:
        # Chemin des fichiers
        base_name = os.path.splitext(os.path.basename(state['pdf_file']))[0]
        output_text_path = os.path.join(output_dir, f"{base_name}.txt")        
        output_json_path = os.path.join(output_dir, f"{base_name}.json")
        
        # Lecture du contenu texte généré précédemment
        with open(output_text_path, "r", encoding="utf-8") as f:
            pdf_content = f.read()

        # Prompt amélioré avec exemples
        system_prompt = """[FR/EN]
En tant qu'archiviste numérique expert, analysez ce document et générez :
1. Titre concis (15 mots max)
2. Date (format: JJ/MM/AAAA)
3. Contexte général (50 mots max)
4. Résumé en 3-4 phrases
5. Entités clés (personnes/organisations/lieux)
6. Type de document (uniquement: Contrat de location/Contrat vacataire/Contrat partenariat académique/Contrat partenariat industriel)

Exemple de réponse VALIDE :
{
    "titre": "Accord de collaboration universitaire",
    "date": "2022-2025",
    "contexte": "Partenariat entre établissements d'enseignement",
    "resume": "Ce document formalise une coopération dans la recherche pédagogique entre l'Université Paris-Saclay et Polytechnique Montréal...",
    "entites_cles": ["Université Paris-Saclay", "Polytechnique Montréal"],
    "type_document": "Contrat partenariat académique"
}

Format JSON STRICT :"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"DOCUMENT:\n{pdf_content[:8000]}\n\nANALYSE:")
        ]

        # Appel du modèle avec contraintes
        response = model_llama.invoke(messages)
        clean_response = response.content.replace("```json", "").replace("```", "").strip()
        
        # Validation renforcée
        try:
            analysis = json.loads(clean_response)
            required_keys = ['titre', 'date', 'contexte', 'resume', 'entites_cles', 'type_document']
            
            if not all(key in analysis for key in required_keys):
                missing = [k for k in required_keys if k not in analysis]
                raise ValueError(f"Champs manquants: {missing}")

            # Nettoyage des valeurs
            analysis.update({
                "source_fichier": os.path.basename(state['pdf_file']),
                "date_analyse": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pages_analysees": state["total_pages"],
                "entites_cles": [e.strip() for e in analysis['entites_cles'] if e.strip()]
            })

        except Exception as e:
            analysis = {
                "erreur": f"Validation échouée: {str(e)}",
                "raw_response": clean_response,
                "source_fichier": os.path.basename(state['pdf_file'])
            }

      


        # Sauvegarde sécurisée
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        return {"next": "__end__", "metadata": analysis}

    except Exception as e:
        return {"error": f"Erreur critique: {str(e)}"}



###LLM function that takes the user query and determines the most relevant source file

def Get_Source_File(AgentState):
    
    message = AgentState["messages"]
    
    user_query = message[-1]

    # Regex pour extraire le nom de fichier avec extension .pdf entre guillemets
    filename_pattern = r'"([^"]+\.pdf)"'
    match = re.search(filename_pattern, user_query)

    if match :
        filename = match.group(1).strip()
        AgentState['relevant_source'] = filename



    return AgentState


###    RAG function that retrieves information based on the query and identified source file.
def Retrieve_Function(AgentState):
    question = AgentState['messages'][-1]

    filename_pattern = r'"([^"]+\.pdf)"'
    match = re.search(filename_pattern, question)

    if match:
        pdf_filename = match.group(1)
        question = question.replace(f'"{pdf_filename}"', '')


    if 'relevant_source' not in AgentState:
        AgentState['response'] = "Désolé, je n'ai pas trouvé le contrat dont vous avez parlé. Pouvez-vous donner plus de détails sur ce document ?"
        return AgentState

    relevant_source = AgentState['relevant_source'].replace(".pdf", ".txt")
    collection_name = relevant_source.replace(".txt", "")

    output_text_path = f"uploads/{relevant_source}"
    chunks = load_or_create_chunks(collection_name, output_text_path)
    bm25_embedder = load_or_create_bm25(collection_name, chunks)
    
    question

    search_results = hybrid_search(
        client=client,
        query=question,
        collection_name=collection_name,
        bm25_embedder=bm25_embedder,
        dense_model=dense_model,
        alpha=0.5,
        top_k=4
    )

    # Conversion des résultats en texte
    context = "\n\n".join([res['payload']['text'] for res in search_results])

    template = """ 
    Répondez à la question en reformulant uniquement les informations explicitement mentionnées , extrait de {source_file}. 
    - Reformulez de manière claire et naturelle.
    - Ne déduisez rien, ne complétez pas les informations manquantes, ne commentez pas le nom du fichier ni la formulation de la question.
    - Si l'information n'est pas présente dans le texte, répondez simplement : "L'information n'est pas précisée dans le texte."
    - Ne disez pas "L'information n'est pas précisée dans le texte."
    - Ne montioner pas dans la reponse que j'ai fornit du contexte .
    Contexte :
    {context}

    Question : {question}
    """




    prompt = ChatPromptTemplate.from_template(template)
    
    # Version simplifiée sans RunnableParallel
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=lambda _: context,
            source_file=lambda _: relevant_source
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    AgentState['response'] = retrieval_chain.invoke({"question": question})
    return AgentState

