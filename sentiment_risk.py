from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import resolve_embed_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
# from dotenv import load_dotenv
import torch
import chromadb
import os
import re

# load_dotenv()
client = Groq()

# --- CONFIGURATION ---
INPUT_FILE = "full_report_parsed.md"

def verbalize_table(table_text: str, context: str) -> str:
    prompt = f"""
Convert the following financial table into concise natural-language sentences. 
Only restate what is explicitly present. 

Context:
{context}

Table:
{table_text}

Return only the description.
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("Table verbalization failed:", e)
        return ""

def process_tables(md_text: str) -> str:
    pattern = r'\|.+\|[\r\n]+\|[-\s:|]+\|(?:[\r\n]+\|.+\|)*'

    def replace_table(match):
        table = match.group(0)

        start = max(0, match.start() - 200)
        context = md_text[start:match.start()].strip()

        verbalized = verbalize_table(table, context)

        return f"\n{verbalized}\n\n{table}\n"

    return re.sub(pattern, replace_table, md_text)

def break_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

keyword_categories = {
            "liquidity": [
                "cash", "liquid", "liquidity", "cash flow", "working capital",
                "current assets", "current liabilities", "cash position"
            ],
            "risk": [
                "risk", "uncertain", "volatility", "exposure", "loss",
                "adverse", "challenge", "threat", "liability"
            ],
            "revenue": [
                "revenue", "sales", "income", "earnings", "profit",
                "margin", "growth", "turnover"
            ],
            "debt": [
                "debt", "loan", "borrowing", "obligation", "leverage",
                "interest", "credit", "liability"
            ],
            "performance": [
                "performance", "efficiency", "productivity", "return",
                "roi", "growth rate", "improvement"
            ]
        }

def find_relevant_keywords(sentences):
    """Find sentences with relevant keywords and categorize them"""
    categorized_sentences = {category: [] for category in keyword_categories}
        
    for sentence in sentences:
        sentence_lower = sentence.lower()
            
        # Check each keyword category
        for category, keywords in keyword_categories.items():
             if any(keyword in sentence_lower for keyword in keywords):
                categorized_sentences[category].append({
                    "sentence": sentence,
                    "category": category,
                    "matched_keywords": [
                        kw for kw in keywords if kw in sentence_lower
                    ]
                })
        
    # Print summary
    for category, items in categorized_sentences.items():
        if items:
            print(f"\n{category.upper()}: Found {len(items)} relevant sentences")
        
    return categorized_sentences

"""Initialize FinBERT model and tokenizer"""
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert", 
    use_safetensors=True
)
labels = ["negative", "neutral", "positive"]
        

def analyze_sentiment_batch(sentences, batch_size=16):
        """Analyze sentiment for multiple sentences (batched for efficiency)"""
        results = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            
            # Process each result in batch
            for j, sent in enumerate(batch):
                sentiment_idx = torch.argmax(probs[j])
                sentiment = labels[sentiment_idx]
                confidence = float(probs[j][sentiment_idx])
                
                results.append({
                    "sentence": sent,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 3)
                })
        
        return results

def process_categorized_sentences(categorized_sentences):
        """Process and analyze sentiment for categorized sentences"""
        category_results = {}
        
        for category, items in categorized_sentences.items():
            if not items:
                continue
            
            print(f"\nAnalyzing {category} sentiment...")
            sentences = [item["sentence"] for item in items]
            
            sentiment_results = analyze_sentiment_batch(sentences)
            
            # Combine with keyword info
            for i, result in enumerate(sentiment_results):
                result["category"] = category
                result["matched_keywords"] = items[i]["matched_keywords"]
            
            category_results[category] = sentiment_results
        
        return category_results

def generate_highlights(category_results):
        """Generate sentiment highlights and summary"""
        summary = {
            "total_sentences": 0,
            "by_category": {},
            "high_confidence_negatives": [],
            "high_confidence_positives": [],
            "risk_level": "LOW"
        }
        
        for category, results in category_results.items():
            if not results:
                continue
            
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            avg_confidence = sum(r["confidence"] for r in results) / len(results)
            
            for r in results:
                sentiment_counts[r["sentiment"]] += 1
                
                # Track high-confidence signals
                if r["sentiment"] == "negative" and r["confidence"] > 0.7:
                    summary["high_confidence_negatives"].append({
                        "category": category,
                        "sentence": r["sentence"][:150] + "...",
                        "confidence": r["confidence"],
                        "keywords": r["matched_keywords"]
                    })
                elif r["sentiment"] == "positive" and r["confidence"] > 0.7:
                    summary["high_confidence_positives"].append({
                        "category": category,
                        "sentence": r["sentence"][:150] + "...",
                        "confidence": r["confidence"],
                        "keywords": r["matched_keywords"]
                    })
            
            summary["by_category"][category] = {
                "total": len(results),
                "sentiment_breakdown": sentiment_counts,
                "avg_confidence": round(avg_confidence, 3)
            }
            summary["total_sentences"] += len(results)
        
        # Determine risk level
        neg_count = len(summary["high_confidence_negatives"])
        if neg_count >= 3:
            summary["risk_level"] = "HIGH"
            summary["risk_reason"] = f"Found {neg_count} high-confidence negative signals"
        elif neg_count >= 1:
            summary["risk_level"] = "MEDIUM"
            summary["risk_reason"] = f"Found {neg_count} high-confidence negative signal(s)"
        else:
            summary["risk_level"] = "LOW"
            summary["risk_reason"] = "No significant negative sentiment detected"
        
        return summary

def run_pipeline():
    # 1. Load Markdown file
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    print(f"Loaded {len(md_text)} characters from {INPUT_FILE}")

    # 2. 
    try:
        # Process tables
        print("Processing tables...")
        md_text = process_tables(md_text)
    except Exception as e:
        print(f"Table processing failed: {e}")
    
    break_sent_list = break_sentences(md_text)

    categorized = find_relevant_keywords(break_sent_list)
    category_results = process_categorized_sentences(categorized)
    highlights = generate_highlights(category_results)
    # print("\n--- RISK SUMMARY ---")
    # print(f"Risk Level: {highlights['risk_level']}")
    # print(f"Reason: {highlights.get('risk_reason','N/A')}")

    return {
        "categorized_sentences": categorized,
        "category_results": category_results,
        "risk_summary": highlights
    }

if __name__ == "__main__":
    results = run_pipeline()