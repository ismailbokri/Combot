from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, flash, send_file
from flask_cors import CORS
import json
import os
import logging
import re
import sqlite3
from langgraph.graph import StateGraph, END, START, Graph
import time
from werkzeug.utils import secure_filename
from langchain_core.messages import HumanMessage
import glob
from dotenv import load_dotenv
from webCrawlerAgent import compiled_graph

TF_ENABLE_ONEDNN_OPTS=0

from db import init_db, create_conversation, add_message, get_conversation, get_all_conversations,search_conversations,delete_conversation
from combot_functions import ChatbotState , understand_query , generate_answer , detect_language , perform_search , generate_title , initialize_data,format_confidence,get_confidence_class,code_summaries,should_rebuild_stores,build_code_stores
from contrat_functions import State , extract_page ,run_llm_ocr ,run_llm_describer ,should_continue , Get_Source_File , Retrieve_Function , load_or_create_chunks , merge_json_files ,count_total_pages, load_or_create_bm25 , load_contracts ,fulltext_search ,index_data ,BM25Embedder
from config import *


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)






##############################################################################
############### Set our First GRAPH (contract processing ) ###################
##############################################################################

graph_builder = StateGraph(State)
graph_builder.add_node("extract_page", extract_page)
graph_builder.add_node("llm", run_llm_ocr)
graph_builder.add_node("describe", run_llm_describer)


# Configuration modifiée des connexions
graph_builder.add_edge("extract_page", "llm")
graph_builder.add_edge("llm", "describe")
graph_builder.add_edge("describe", END)
graph_builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "continue": "extract_page",
        "end": "describe"
    }
)
graph_builder.set_entry_point("extract_page")
graph_1 = graph_builder.compile()


##############################################################################
############### Set our Second GRAPH (contract chatbot) ###################
##############################################################################



workflow4 = Graph()
workflow4.add_node("DocumentSelector", Get_Source_File)
workflow4.add_node("RAGtool", Retrieve_Function)
workflow4.add_edge('DocumentSelector', 'RAGtool')
workflow4.set_entry_point("DocumentSelector")
workflow4.set_finish_point("RAGtool")
graph_2 = workflow4.compile()

##############################################################################
############### Set our Third GRAPH (contract chatbot) ###################
##############################################################################


workflow = StateGraph(ChatbotState)
workflow.add_node("understand_query", understand_query)
workflow.add_node("perform_search", perform_search)
workflow.add_node("generate_answer", generate_answer)
workflow.set_entry_point("understand_query")
workflow.add_edge("understand_query", "perform_search")
workflow.add_edge("perform_search", "generate_answer")
workflow.add_edge("generate_answer", END)
graph = workflow.compile()


##############################################################################
######################## Application Flask ###################################
##############################################################################


# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400
    query = data["query"].strip()
    conversation_id = data.get("conversation_id")
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Track processing time for analytics
    processing_start = time.time()
    
    if conversation_id:
        conn = sqlite3.connect("conversations.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "Conversation not found"}), 404
        conn.close()
    else:
        title = generate_title(query)
        conversation_id = create_conversation(title)

    add_message(conversation_id, "user", query)
    
    # Log the beginning of query processing
    logging.info(f"Processing query: '{query}' for conversation {conversation_id}")

    initial_state = {
        "query": query,
        "reasoning_steps": [],
        "search_results": [],
        "final_answer_en": "",
        "final_answer_fr": "",
        "sources": [],
        "thinking_time": 0.0
    }
    
    try:
        # Run the agent workflow
        final_state = graph.invoke(initial_state)
        logging.info(f"Agent workflow completed in {time.time() - processing_start:.2f}s")
    except Exception as e:
        logging.error(f"Agent workflow failed: {str(e)}")
        # Fallback state with error message
        final_state = {
            "query": query,
            "reasoning_steps": [{"step": "error", "text": f"Processing error: {str(e)}"}],
            "search_results": [],
            "final_answer_en": "An error occurred while processing your request. Please try again later.",
            "final_answer_fr": "Une erreur s'est produite lors du traitement de votre demande. Veuillez réessayer plus tard.",
            "sources": [],
            "thinking_time": 0.0
        }

    # Build HTML for assistant's response with enhanced reasoning display
    reasoning_block = ""
    if final_state["reasoning_steps"]:
        # Extract and organize thinking steps for better presentation
        thinking_steps = []
        for step in final_state["reasoning_steps"]:
            if step["step"] == "thinking" or step["step"] == "reasoning_en" or step["step"] == "reasoning_fr":
                thinking_steps.append({
                    "title": "Chain of Thought Reasoning",
                    "content": step["text"]
                })      
    # Deduplicate thinking steps first
    seen = set()
    unique_steps = []
    for step in thinking_steps:
        content = step["content"].strip()
        if content not in seen:
            seen.add(content)
            unique_steps.append(step)

    thinking_steps = unique_steps

    # Then render them
    if thinking_steps:
        thinking_sections = "".join([
            f"""
            <div class="thinking-section">
            <h4>{step['title']}</h4>
            <div class="thinking-content">{step['content'].replace('/n', '<br>')}</div>
            </div>
            """
            for step in thinking_steps
        ])

        reasoning_block = f"""
        <details class="thinking-block">
        <summary>Show Chain of Thought Analysis</summary>
        <div class="thinking-container">
            {thinking_sections}
        </div>
        </details>
        """

        # Format the final answer based on language
        lang = detect_language(query)
        final_answer = final_state['final_answer_fr'] if lang == 'fr' else final_state['final_answer_en']
        
        # Apply proper formatting to the final answer
        final_answer_formatted = final_answer.replace('\n', '<br>')
        pattern = r"(.*?)(\*\*(Réponse|Answer):\*\*)"

        match = re.search(pattern, final_answer_formatted)
        if match:
            final_answer_formatted = match.group(0)

        final_answer_block = f"""
        <div class="final-answer">
        {final_answer_formatted}
        </div>
        """

        # Enhanced sources display with confidence scores
        sources_html = ""
        if final_state["sources"]:
            # Sort sources by confidence score, highest first
            sorted_sources = sorted(final_state["sources"], key=lambda x: x.get('confidence', 0), reverse=True)
            
            source_items = "".join([
                f"""
                <li class="source-item {get_confidence_class(src.get('confidence', 0))}">
                <div class="source-info">
    <span class="source-name">
    <strong>{src['article']} with Confidence of {src['confidence'] * 100:.2f}%</strong>
    </span>
                    <span class="confidence-badge">{format_confidence(src.get('confidence', 0))}</span>
                </div>
                <a href="#" class="source-link" onclick="showArticlePopup('{src['article']}', `{src['text'].replace('`', '/`')}`, '{format_confidence(src.get('confidence', 0))}'); return false;">
                    View Content
                </a>
                </li>
                """
                for src in sorted_sources
            ])
            
            sources_html = f"""
            <details class="sources-toggle">
            <summary>Show Legal Sources ({len(sorted_sources)})</summary>
            <ul class="sources-list">
                {source_items}
            </ul>
            </details>
            """

        assistant_html = reasoning_block + final_answer_block + sources_html
        
        # Add CSS for enhanced display
        assistant_html = f"""
        <style>
        .thinking-block {{
            margin-top: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .thinking-container {{
            padding: 1rem;
            background-color: #f9f9f9;
        }}
        .thinking-section {{
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #eaeaea;
        }}
        .thinking-section h4 {{
            margin-top: 0;
            color: #2c5282;
        }}
        .final-answer {{
            padding: 1rem;
            background-color: #f0f7ff;
            border-left: 4px solid #2c5282;
            margin: 1rem 0;
            border-radius: 4px;
        }}
        .sources-toggle {{
            margin-top: 1rem;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .sources-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .source-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #eaeaea;
        }}
        .source-item:last-child {{
            border-bottom: none;
        }}
        .source-info {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .confidence-badge {{
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: bold;
        }}
        .high-confidence {{
            background-color: #c6f6d5;
        }}
        .medium-confidence {{
            background-color: #fefcbf;
        }}
        .low-confidence {{
            background-color: #fed7d7;
        }}
        .source-link {{
            padding: 0.25rem 0.75rem;
            background-color: #4299e1;
            color: white;
            border-radius: 4px;
            text-decoration: none;
            font-size: 0.875rem;
        }}
        .source-link:hover {{
            background-color: #3182ce;
        }}
        </style>
        {assistant_html}
        """
        
        # ADD CODE IDENTIFIER HERE
        if "code_name" in final_state and final_state["code_name"]:
            code_name = final_state["code_name"]
            code_message = f"""
            <div class="code-identifier">
                <span class="code-label">Legal Code:</span> 
                <span class="code-name">{code_name}</span>
                <span class="similarity-score">Relevance: {final_state.get("code_similarity", 0) * 100:.1f}%</span>
            </div>
            """
            
            # Add CSS for the code identifier
            code_identifier_style = """
            <style>
            .code-identifier {
                margin-bottom: 1rem;
                padding: 0.75rem;
                border-radius: 8px;
                background-color: #f0f7ff;
                border-left: 4px solid #3182ce;
                font-size: 0.95rem;
            }
            .code-label {
                font-weight: bold;
                color: #2c5282;
                margin-right: 0.5rem;
            }
            .code-name {
                font-weight: bold;
                color: #2c5282;
            }
            .similarity-score {
                margin-left: 1rem;
                color: #4a5568;
                font-size: 0.85rem;
            }
            </style>
            """
            
            # Add the code message to the beginning of the HTML
            assistant_html = code_identifier_style + code_message + assistant_html
        
        # ADD CODE IDENTIFIER HERE - after all HTML is generated but before add_message
    if "code_name" in final_state and final_state["code_name"]:
        code_name = final_state["code_name"]
        code_similarity = final_state.get("code_similarity", 0)
        
        # Get the code summary
        code_summary = code_summaries.get(code_name, "")
        
        # Determine the class of relevance
        relevance_class = "high-relevance" if code_similarity > 0.7 else "medium-relevance" if code_similarity > 0.4 else "low-relevance"
        
        add_message(conversation_id, "assistant", assistant_html)

        # Calculate total processing time
        total_processing_time = time.time() - processing_start
        
        # Prepare response metrics
        metrics = {
            "thinking_time": int(final_state["thinking_time"]),
            "total_processing_time": int(total_processing_time),
            "search_result_count": len(final_state["sources"]),
            "reasoning_steps_count": len([step for step in final_state["reasoning_steps"] if step["step"] in ("thinking", "reasoning_en", "reasoning_fr")]),
        }
        
        logging.info(f"Request processed in {total_processing_time:.2f}s (thinking: {final_state['thinking_time']:.2f}s)")
        
        return jsonify({
            "conversation_id": conversation_id,
            "title": title if not data.get("conversation_id") else None,
            "assistant_html": assistant_html,
            "sources": final_state["sources"],
            "metrics": metrics
        })

@app.route("/get_conversations", methods=["GET"])
def get_conversations():
    return jsonify(get_all_conversations())

@app.route('/list_codes', methods=['GET'])
def list_codes():
    """List all available legal codes with their summaries."""
    available_codes = []
    for code_name, summary in code_summaries.items():
        available = code_name in code_stores if 'code_stores' in globals() else False
        available_codes.append({
            'code_name': code_name,
            'summary': summary,
            'available': available
        })
    
    return jsonify({
        'codes': available_codes
    })

@app.route('/article/<code>/<article_id>', methods=['GET'])
def get_article_content(code, article_id):
    """Endpoint to retrieve the content of a specific article."""
    try:
        # Look for the article in the code store
        if 'code_stores' in globals() and code in code_stores:
            # Try to find the specific article
            store = code_stores[code]
            # Basic approach - this can be improved based on your data structure
            article_query = f"Article {article_id}"
            docs = store.similarity_search(article_query, k=1)
            
            if docs:
                return jsonify({
                    'code': code,
                    'article_id': article_id,
                    'content': docs[0].page_content,
                    'metadata': docs[0].metadata
                })
        
        # If we get here, we couldn't find the article
        # Fallback to simulated content like in the original code
        article_content = f"Contenu de l'article {article_id} du code {code}.\n\n"
        
        # Add simulated content based on code type
        if code == "loi_defense_contre_pratiques_deloyales_importation":
            article_content += "Cet article concerne les pratiques déloyales d'importation et définit les mesures à prendre pour protéger le marché national."
        elif code == "loi_relative_commerce_exterieur":
            article_content += "Cet article précise les conditions du commerce extérieur et les obligations des importateurs et exportateurs."
        elif code == "loi_relative_Startups":
            article_content += "Cet article détaille les avantages fiscaux et financiers accordés aux startups labellisées."
        else:
            article_content += "Détails relatifs à cet article du code juridique tunisien."
        
        return jsonify({
            'code': code,
            'article_id': article_id,
            'content': article_content
        })
        
    except Exception as e:
        logging.error(f"Error retrieving article content: {e}")
        return jsonify({
            'error': f"Erreur lors de la récupération de l'article: {str(e)}"
        }), 500

@app.route("/get_conversation/<int:conversation_id>", methods=["GET"])
def get_conversation_route(conversation_id):
    conversation = get_conversation(conversation_id)
    if conversation:
        return jsonify(conversation)
    return jsonify({"error": "Conversation not found"}), 404

@app.route("/delete_conversation/<int:conversation_id>", methods=["DELETE"])
def delete_conversation_route(conversation_id):
    delete_conversation(conversation_id)
    return jsonify({"success": True})

@app.route("/search_conversations", methods=["POST"])
def search_conversations_route():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify(get_all_conversations())
    return jsonify(search_conversations(query))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/js/article_popup.js')
def article_popup_js():
    js_content = """
function showArticlePopup(articleTitle, articleContent, confidenceLabel) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('articleModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'articleModal';
        modal.className = 'article-modal';
        document.body.appendChild(modal);
        
        // Add styles if not already in CSS
        if (!document.getElementById('article-modal-style')) {
            const style = document.createElement('style');
            style.id = 'article-modal-style';
            style.textContent = `
                .article-modal {
                    display: none;
                    position: fixed;
                    z-index: 1000;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    overflow: auto;
                    background-color: rgba(0,0,0,0.4);
                    animation: fadeIn 0.3s ease-out;
                }
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                .article-modal-content {
                    background-color: #fefefe;
                    margin: 5% auto;
                    padding: 24px;
                    border: 1px solid #e2e8f0;
                    width: 90%;
                    max-width: 800px;
                    border-radius: 12px;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    max-height: 80vh;
                    overflow-y: auto;
                    animation: slideIn 0.3s ease-out;
                }
                @keyframes slideIn {
                    from { transform: translateY(-20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
                .article-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    margin-bottom: 16px;
                    padding-bottom: 16px;
                    border-bottom: 1px solid #e2e8f0;
                }
                .article-title-container {
                    flex: 1;
                }
                .article-close {
                    color: #a0aec0;
                    font-size: 24px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: color 0.2s;
                    background: none;
                    border: none;
                    padding: 0;
                    height: 24px;
                    line-height: 24px;
                    margin-left: 16px;
                }
                .article-close:hover {
                    color: #2d3748;
                }
                .article-title {
                    margin: 0 0 4px 0;
                    color: #2d3748;
                    font-size: 20px;
                    font-weight: 600;
                }
                .article-subtitle {
                    color: #718096;
                    font-size: 14px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .confidence-badge {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 9999px;
                    font-size: 12px;
                    font-weight: 500;
                }
                .confidence-high {
                    background-color: #c6f6d5;
                    color: #22543d;
                }
                .confidence-medium {
                    background-color: #fefcbf;
                    color: #744210;
                }
                .confidence-low {
                    background-color: #fed7d7;
                    color: #822727;
                }
                .article-body {
                    white-space: pre-wrap;
                    line-height: 1.6;
                    color: #4a5568;
                    font-size: 16px;
                    padding: 8px 0;
                }
                .article-actions {
                    margin-top: 16px;
                    display: flex;
                    justify-content: flex-end;
                    gap: 8px;
                }
                .article-action-button {
                    padding: 8px 16px;
                    background-color: #edf2f7;
                    color: #2d3748;
                    border: none;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }
                .article-action-button:hover {
                    background-color: #e2e8f0;
                }
                .highlight {
                    background-color: #fef3c7;
                    padding: 2px 0;
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    // Get confidence class
    let confidenceClass = "confidence-medium";
    if (confidenceLabel && confidenceLabel.includes("High")) {
        confidenceClass = "confidence-high";
    } else if (confidenceLabel && confidenceLabel.includes("Low")) {
        confidenceClass = "confidence-low";
    }
    
    // Format article content to highlight legal terms
    const formattedContent = formatLegalContent(articleContent);

    // Update modal content
    modal.innerHTML = `
        <div class="article-modal-content">
            <div class="article-header">
                <div class="article-title-container">
                    <h3 class="article-title">${articleTitle}</h3>
                    <div class="article-subtitle">
                        <span class="confidence-badge ${confidenceClass}">${confidenceLabel || 'Source'}</span>
                    </div>
                </div>
                <button class="article-close">&times;</button>
            </div>
            <div class="article-body">${formattedContent}</div>
            <div class="article-actions">
                <button class="article-action-button" onclick="copyToClipboard('${articleTitle}', this)">Copy Article Reference</button>
            </div>
        </div>
    `;

    // Show modal with animation
    modal.style.display = 'block';

    // Add close functionality
    const closeBtn = modal.querySelector('.article-close');
    closeBtn.onclick = function() {
        closeArticleModal();
    }

    // Close when clicking outside the modal
    window.onclick = function(event) {
        if (event.target == modal) {
            closeArticleModal();
        }
    }
    
    // Close when pressing Escape
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && modal.style.display === 'block') {
            closeArticleModal();
        }
    });
}

function closeArticleModal() {
    const modal = document.getElementById('articleModal');
    if (modal) {
        // Add closing animation
        modal.style.opacity = '0';
        modal.style.transform = 'translateY(-10px)';
        
        // Remove after animation completes
        setTimeout(() => {
            modal.style.display = 'none';
            modal.style.opacity = '1';
            modal.style.transform = 'translateY(0)';
        }, 200);
    }
}

function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        const originalText = button.textContent;
        button.textContent = "Copied!";
        button.style.backgroundColor = "#c6f6d5";
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.backgroundColor = "";
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy: ', err);
        button.textContent = "Failed to copy";
        button.style.backgroundColor = "#fed7d7";
        
        setTimeout(() => {
            button.textContent = "Copy Article Reference";
            button.style.backgroundColor = "";
        }, 2000);
    });
}

function formatLegalContent(content) {
    if (!content) return '';
    
    // Highlight article numbers
    let formatted = content.replace(/\\b(Article\\s+\\d+)\\b/g, '<strong>$1</strong>');
    
    // Highlight legal terms (simplified version)
    const legalTerms = [
        'loi', 'décret', 'circulaire', 'règlement', 'jurisprudence', 'tribunal', 'cour', 
        'justice', 'jugement', 'contentieux', 'procédure', 'avocat', 'responsabilité', 
        'contrat', 'obligation', 'droit', 'propriété', 'civil', 'pénal', 'fiscal', 
        'administratif', 'commercial', 'sociale', 'travail', 'constitution',
        'law', 'decree', 'circular', 'regulation', 'court', 'justice', 'judgment', 
        'litigation', 'procedure', 'lawyer', 'attorney', 'liability', 'contract', 
        'obligation', 'right', 'property', 'civil', 'criminal', 'tax', 'administrative', 
        'commercial', 'social', 'labor'
    ];
    
    // Create a regex pattern for all legal terms with word boundaries
    const pattern = new RegExp('\\\\b(' + legalTerms.join('|') + ')\\\\b', 'gi');
    formatted = formatted.replace(pattern, '<span class="highlight">$1</span>');
    
    return formatted;
}
    """
    return js_content, 200, {'Content-Type': 'application/javascript'}

@app.route('/rebuild_stores', methods=['GET'])
def rebuild_stores_route():
    try:
        force_rebuild = request.args.get('force', 'false').lower() == 'true'
        
        if force_rebuild or should_rebuild_stores():
            # Force rebuild of stores
            stores_dir = config.get("stores_directory", "stores")
            
            if os.path.exists(stores_dir):
                # Only remove specific subdirectories, keeping build_metadata.json
                for item in os.listdir(stores_dir):
                    item_path = os.path.join(stores_dir, item)
                    # Skip the metadata file when cleaning up
                    if item != "build_metadata.json" and os.path.exists(item_path):
                        if os.path.isdir(item_path):
                            import shutil
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
            
            # Build the stores
            success = build_code_stores()
            
            # Reload the data
            initialize_data()
            
            if success:
                message = "Forced rebuild completed successfully." if force_rebuild else "Rebuild due to changes completed successfully."
            else:
                message = "Rebuild encountered errors, check the logs."
                
            return jsonify({
                "success": success,
                "message": message
            })
        else:
            return jsonify({
                "success": True,
                "message": "No rebuild needed. Stores are up to date."
            })
    except Exception as e:
        logging.error(f"Error rebuilding stores: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500






























########################################################################
@app.route('/contrat', methods=['GET', 'POST'])
def index():
    already_processed = False
    pdf_text = None
    
    if request.method == 'POST':
        # Vérification de base du formulaire
        if 'file' not in request.files:
            flash("Aucun fichier sélectionné")
            return redirect(request.url)
        
        file = request.files['file']
        
        # Validation du fichier
        if file.filename == '':
            flash("Veuillez sélectionner un fichier")
            return redirect(request.url)
            
        if not file.filename.lower().endswith('.pdf'):
            flash("Seuls les fichiers PDF sont acceptés")
            return redirect(request.url)

        # Création sécurisée du chemin
        upload_dir = UPLOAD_FOLDER 
        os.makedirs(upload_dir, exist_ok=True)  # Crée le dossier si inexistant
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        
        try:
            # Sauvegarde temporaire du PDF
            file.save(filepath)
            print(f"Fichier sauvegardé avec succès à : {filepath}")  # Debug
            
        except Exception as e:
            flash(f"Erreur lors de l'enregistrement : {str(e)}")
            return redirect(request.url)

        # Vérification de l'existence du traitement
        base_name = os.path.splitext(filename)[0]
        text_path = os.path.join(upload_dir, f"{base_name}.txt")
        
        if os.path.exists(text_path):
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    pdf_text = f.read()
                return render_template('contrat.html',
                                    pdf_text=pdf_text,
                                    already_processed=True)
            except Exception as e:
                flash(f"Erreur de lecture du fichier existant : {str(e)}")

        # Traitement du nouveau fichier
        try:
            initial_state = {
                "messages": [HumanMessage(content="Démarrage du traitement OCR")],
                "page_number": 1,
                "total_pages": count_total_pages(filepath),
                "pdf_file": filepath
            }

            # Exécution du pipeline
            for event in graph_1.stream(initial_state):
                if "error" in event:
                    flash(f"Erreur de traitement : {event['error']}")
                    break
                
                if event.get("page_number", 0) > initial_state["total_pages"]:
                    break

            # Récupération du résultat
            output_text_path = os.path.join("uploads", f"{base_name}.txt")

            if os.path.exists(output_text_path):

                with open(output_text_path, "r", encoding="utf-8") as f:
                    pdf_text = f.read()
                    merge_json_files()

            else:
                flash("Aucun résultat généré")

        finally:
            # Nettoyage sécurisé
            if os.path.exists(filepath):
                try:
                    #### Indexation
                    collection_name = base_name
                    print(f"\n{'='*50}\nDémarrage avec la collection: {collection_name}\n{'='*50}")
                        
                        # Chargement ou création des chunks
                    chunks = load_or_create_chunks(collection_name, output_text_path)
                    print(f"Nombre de chunks chargés: {len(chunks)}")
                        
                    bm25_embedder = load_or_create_bm25(collection_name, chunks)
                        
                        # Indexation si nécessaire
                    index_data(client, chunks, collection_name, bm25_embedder, dense_model)
                    print(f"Fichier temporaire supprimé : {filepath}")  # Debug
                except Exception as e:
                    print(f"Erreur suppression fichier : {str(e)}")

    return render_template('contrat.html',
                         pdf_text=pdf_text,
                         already_processed=already_processed)

@app.route('/chat')
def chat():
    contracts = load_contracts()
    return render_template("chat_contrat.html", contracts=contracts)

@app.route('/pdf/<filename>')
def serve_pdf(filename):
    safe_filename = secure_filename(filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    if not os.path.exists(pdf_path):
        return "Fichier non trouvé", 404
    
    return send_file(pdf_path, as_attachment=False)

@app.route('/search-page')
def search_page():
    return render_template('search.html')

@app.route('/legal-chat')
def index_page():
    return render_template('index.html')

# Modifier la route existante pour l'API de recherche
@app.route('/api/search')
def search_documents():
    query = request.args.get('q', '').lower()
    exact_match = request.args.get('exact', 'false') == 'true'
    results = []
    
    json_files = glob.glob(os.path.join(UPLOAD_FOLDER, '*.json'))
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if fulltext_search(query, data, exact_match):
                    results.append(data)
            except json.JSONDecodeError:
                continue
    
    return jsonify(results[:20])

@app.route('/document/<filename>')
def get_document(filename):
    safe_filename = secure_filename(filename)
    json_path = os.path.join(UPLOAD_FOLDER, f"{os.path.splitext(safe_filename)[0]}.json")
    
    if not os.path.exists(json_path):
        return jsonify({"error": "Document non trouvé"}), 404
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pdf_exists = os.path.exists(
        os.path.join(UPLOAD_FOLDER, data['source_fichier'])
    )
    
    return jsonify({**data, "pdf_exists": pdf_exists})


@app.route("/get_response", methods=["POST"])
def get_response():
    user_query = request.form["user_query"]
    contracts = load_contracts()
    
    inputs = {"messages": [user_query]}
    ai_response = ""
    
    for output in graph_2.stream(inputs):
        if 'RAGtool' in output:
            ai_response = output['RAGtool'].get('response', "No response found")
    
    return render_template("chat_contrat.html", 
                         ai_response=ai_response,
                         contracts=contracts,
                         user_query=user_query)

@app.route('/test-mehdi')
def test():
    return render_template('jort.html')

@app.route("/check-updates", methods=["GET"])
def check_updates():
    # Simulated data (you can replace this with your real logic)
    initial_state = {"input": "Check for new JORTS and generate a summary."}
    result = compiled_graph.invoke(initial_state)
    print(result["updated_summaries"])
    return jsonify(result["updated_summaries"])
    # laws = [
    #     {"title": "Loi n° 2024-01", "summary": "Résumé de la loi sur les sociétés."},
    #     {"title": "Décret n° 2024-15", "summary": "Décret relatif au blanchiment d'argent."},
    #     {"title": "Arrêté n° 2024-32", "summary": "Arrêté concernant le change monétaire."}
    # ]
    # return jsonify(laws)



if __name__ == "__main__":
    # Initialize directories
    stores_dir = config.get("stores_directory", "stores")
    os.makedirs(stores_dir, exist_ok=True)
    
    # Check if rebuild is needed
    if should_rebuild_stores():
        print("Changes detected - rebuilding stores...")
        build_code_stores()
    else:
        print("No changes detected - using existing stores")
    
    # Verify stores exist
    if os.path.exists(stores_dir):
        store_files = os.listdir(stores_dir)
        print(f"Stores directory contains: {', '.join(store_files)}")
    
    # Continue with initialization
    init_db()
    initialize_data()
    app.run(debug=False, host="0.0.0.0", port=5000)
