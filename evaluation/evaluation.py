import json
import evaluate

# Load merged data
with open(f'evaluation/merged_question_answers.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract lists of generated and reference answers
predictions = [entry['generated_answer'] for entry in data]
references = [entry['reference_answer'] for entry in data]

# Initialize evaluation metrics
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
bertscore = evaluate.load('bertscore')

# Compute ROUGE
rouge_scores = rouge.compute(predictions=predictions, references=references)

# Compute BLEU
# BLEU expects references as list of list of tokens
bleu_scores = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

# Compute BERTScore (uses multilingual BERT)
bertscore_scores = bertscore.compute(predictions=predictions, references=references, lang="fr")

# Print results
print("\n--- Evaluation Results ---")
print("ROUGE Scores:")
for k, v in rouge_scores.items():
    print(f"{k}: {v:.4f}")

print(f"\nBLEU Score: {bleu_scores['bleu']:.4f}")

print(f"\nBERTScore (F1 average): {sum(bertscore_scores['f1']) / len(bertscore_scores['f1']):.4f}")








"""
--- Evaluation Results ---
ROUGE Scores:
rouge1: 0.4960
rouge2: 0.3422
rougeL: 0.4359


BLEU Score: 0.1923

BERTScore (F1 average): 0.7955

"""







########################################################################################################

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation) #####
# Measures how much the generated answer overlaps with the reference answer, using:
# ROUGE-1: Unigram (word) overlap
# ROUGE-2: Bigram (2-word sequence) overlap
# ROUGE-L: Longest common subsequence (LCS)
# ✅ Best for:
# Checking if the output reuses the correct words or phrases

#### BLEU (Bilingual Evaluation Understudy) #####
# How many n-grams in the generated output match the reference 
# ❌ Weakness:
# Too harsh on paraphrasing
# Doesn't understand meaning, only exact n-gram overlap

#### BERTScore #####
# Uses BERT embeddings to measure semantic similarity between generated and reference answers
# The algorithm computes cosine similarity precision, recall, and F1-score
# ✅ Best for:
# Capturing semantic meaning, not just exact match


##############################################################################################################

