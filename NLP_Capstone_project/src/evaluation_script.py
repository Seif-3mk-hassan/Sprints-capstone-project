import json
import requests
from rouge_score import rouge_scorer
from tqdm import tqdm

BACKEND_URL = "http://127.0.0.1:8001/generate"

TEST_FILE = "NLP_Capstone_project/data/test_subset.json"
OUTPUT_FILE = "NLP_Capstone_project/outputs/evaluation_results.json"

PROMPT_STYLE = "few-shot"
USE_RAG = True
K = 3


def load_test_data():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_reply(issue):
    payload = {
        "issue": issue,
        "use_rag": USE_RAG,
        "prompt_style": PROMPT_STYLE,
        "k": K,
        "return_retrieval": False  
    }

    try:
        response = requests.post(BACKEND_URL, json=payload)
        if response.status_code != 200:
            return ""

        return response.json().get("reply", "")
    except:
        return ""


def evaluate():
    data = load_test_data()

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    results = []
    scores = []

    for item in tqdm(data):
        issue = item["customer_issue"]
        reference = item["reference_reply"]

        generated = generate_reply(issue)

        score = scorer.score(reference, generated)["rougeL"].fmeasure
        scores.append(score)

        results.append({
            "issue": issue,
            "reference": reference,
            "generated": generated,
            "rougeL": score
        })

    avg_score = sum(scores) / len(scores) if scores else 0

    output = {
        "average_rougeL": avg_score,
        "num_samples": len(results),
        "config": {
            "prompt_style": PROMPT_STYLE,
            "use_rag": USE_RAG,
            "k": K
        },
        "details": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Evaluation completed. Avg ROUGE-L: {avg_score:.4f}")


if __name__ == "__main__":
    evaluate()