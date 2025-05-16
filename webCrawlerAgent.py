import requests
import random
import time
from bs4 import BeautifulSoup
from datetime import datetime
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry
import locale
import re

from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

# Setup requests session
session = requests.Session()
retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
]
headers = {"User-Agent": random.choice(user_agents)}
web_url = "https://9anoun.tn/fr/kb/jorts"


# --------- Tools ---------

@tool
def check_new_jorts_tool() -> dict:
    """Check if there are new JORTS published and return their dates and links."""
    try:
        r = session.get(web_url, headers=headers)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
    except RequestException as e:
        print(f"Failed to access {web_url}: {e}")
        return {"new_jorts_table": []}

    jort_table = soup.find('div', class_='Pd Ki Fc sH YJ QE')
    jorts = jort_table.find_all('div', class_='Ee')
    jorts_to_check = []
    for jort in jorts:
        date = jort.find('div', class_='Or Iq').text.strip()
        link = jort.find('a').get('href')
        if date and link:
            jorts_to_check.append((date, link))

    try:
        with open('last_update.txt', 'r', encoding='utf-8') as f:
            last_update_str = f.read().strip()
    except FileNotFoundError:
        last_update_str = "01 janvier 2000"

    last_update_date = datetime.strptime(last_update_str, "%d %B %Y")

    new_jorts_table = []
    for date_str, link in jorts_to_check:
        date_obj = datetime.strptime(date_str, "%d %B %Y")
        if date_obj > last_update_date:
            new_jorts_table.append((date_str, link))

    return {"new_jorts_table": new_jorts_table}

@tool
def fetch_jorts_data_tool(new_jorts_table: list) -> dict:
    """Fetch articles from new JORTS links."""
    data_collected = []
    for date_str, link in new_jorts_table:
        try:
            r_jort = session.get(link, headers=headers)
            r_jort.raise_for_status()
            soup_jort = BeautifulSoup(r_jort.content, 'html.parser')
        except RequestException as e:
            print(f"Failed to access {link}: {e}")
            continue

        subject_table = soup_jort.find('div', class_='xd Cj')
        contents = subject_table.find_all('a', class_='Jd Iq Pr Lw lx jk Lo zo')
        article_links = [content.get('href') for content in contents]

        for article_link in article_links:
            try:
                r_article = session.get(article_link, headers=headers)
                r_article.raise_for_status()
                soup_article = BeautifulSoup(r_article.content, 'html.parser')
            except RequestException as e:
                print(f"Failed to access {article_link}: {e}")
                continue

            title = soup_article.find('h1', class_='xd QG EL xq Sq').text.strip()
            data = soup_article.find('div', class_='yd Kc Hq Bu').text.strip()
            data_collected.append({"title": title, "content": data})

            time.sleep(random.uniform(2, 4))

    return {"articles": data_collected}

@tool
def filter_mise_a_jour_tool(summaries: list) -> dict:
    """
    Update the 'relevency' field to 'yes' for summaries indicating updates ('mise à jour'),
    excluding those explicitly stating 'pas de mise à jour'.
    """
    for s in summaries:
        text = s["summary"].lower()
        if "mise a jour" in text or "mise à jour" in text:
            if re.search(r"\bpas de mise (a|à) jour\b", text):
                continue
            s["relevency"] = "yes"
    return {"updated_summaries": summaries}

# # --------- LLM Node ---------

def generate(state):
    """Generate a summary for each collected JORTS article one-by-one."""
    print("---GENERATE---")
    articles = state["articles"]

    # Setup Prompt
    prompt = PromptTemplate(
        template="""Voici un extrait d'un journal officiel:

TITRE: {title}

CONTENU: {content}

gardes juste les articles qui on une relation avec le code de change,Le Code de Blanchiment d'Argent,Le Code des Sociétés et du Commerce,Le Code des Obligations et des Contrats

1. Le Code de Change tunisien régit les opérations de change, les transactions en devises et les transferts financiers internationaux, en encadrant la convertibilité du dinar et les obligations des résidents et non-résidents.

2. Le Code de Blanchiment d'Argent définit les infractions liées au blanchiment de capitaux, fixe les mesures de prévention, de détection et de répression, et impose aux institutions financières et non financières des obligations de vigilance.

3. Le Code des Sociétés et du Commerce organise la création, le fonctionnement et la dissolution des sociétés commerciales en Tunisie, fixe les règles du commerce général, des contrats commerciaux et des procédures de faillite.

4. Le Code des Obligations et des Contrats constitue la base du droit civil tunisien en matière contractuelle, établissant les principes généraux des obligations, la formation, l'exécution et l'extinction des contrats.


garder les information comme elles sont no modifie rien dans le cantenu
si il n y pas de contenue sur ces code repond avec "pas de mise a jours"
si il  y a de contenue sur ces code repond avec "mise a jours"
""",
        input_variables=["title", "content"]
    )

    # Setup LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  
        temperature=0,
        max_tokens=None,
        api_key="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"
    )

    rag_chain = prompt | llm | StrOutputParser()

    summaries = []

    for article in articles:
        title = article["title"]
        content = article["content"]

        try:
            # Invoke LLM per article
            response = rag_chain.invoke({"title": title, "content": content})
            print(response)
            summaries.append({
                "title": title,
                "summary": response,
                "content": content,
                "relevency": "no"
            })
        except Exception as e:
            print(f"Error processing article '{title}': {e}")
            continue

    return {"summaries": summaries}


# --------- Node Functions ---------

def run_check_new_jorts(state: dict):
    """Wrapper function to call the check_new_jorts_tool"""
    tool_result = check_new_jorts_tool.invoke({})  # Pass empty dict as tool doesn't need input
    return {**state, **tool_result}

def run_fetch_jorts_data(state: dict):
    """Wrapper function to call the fetch_jorts_data_tool"""
    tool_result = fetch_jorts_data_tool.invoke({"new_jorts_table": state["new_jorts_table"]})
    return {**state, **tool_result}

def run_generate(state: dict):
    """Wrapper function to call the generate function"""
    result = generate(state)
    return {**state, **result}

def run_filter_updates(state: dict):
    tool_result = filter_mise_a_jour_tool.invoke({"summaries": state["summaries"]})
    return {**state, **tool_result}

# --------- LangGraph Setup ---------

graph = StateGraph(dict)  # Using plain dict for state

graph.add_node("check_new_jorts", run_check_new_jorts)
graph.add_node("fetch_jorts_data", run_fetch_jorts_data)
graph.add_node("generate", run_generate)
graph.add_node("filter_updates", run_filter_updates)

# Edges
graph.add_edge("check_new_jorts", "fetch_jorts_data")
graph.add_edge("fetch_jorts_data", "generate")
graph.add_edge("generate", "filter_updates")
graph.add_edge("filter_updates", END)

# Set EntryPoint
graph.set_entry_point("check_new_jorts")

# Compile
compiled_graph = graph.compile()

# Execute
# initial_state = {"input": "Check for new JORTS and generate a summary."}
# result = compiled_graph.invoke(initial_state)
# print(result["updated_summaries"])