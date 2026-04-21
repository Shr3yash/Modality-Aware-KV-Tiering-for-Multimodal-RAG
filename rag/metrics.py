import json
import re
from collections import Counter
from typing import Dict, List, Optional
from openai import OpenAI

JUDGE_PROMPT = """
    You are evaluating a multimodal RAG system output.
    Judge two things:
    1. Correctness: whether the predicted answer is semantically correct with respect to the gold answer.
    2. Faithfulness: whether the predicted answer is supported by the provided evidence.

    Rules:
    - Focus on semantic equivalence, not exact wording.
    - Ignore citation formatting like [text3] or [image2].
    - Ignore extra explanation if the core answer is correct.
    - Treat numerically equivalent answers as correct only when they mean the same thing.
    - Be careful to distinguish percent from percentage point, and absolute from relative change.
    - If the prediction includes the correct answer plus unsupported extra claims, correctness may still be high but faithfulness should be lower.

    Correctness score rubric:
    5 = fully correct
    4 = correct but slightly imprecise
    3 = partially correct
    2 = mostly incorrect
    1 = incorrect but related
    0 = completely incorrect

    Faithfulness score rubric:
    5 = fully supported by evidence
    4 = mostly supported with minor unsupported detail
    3 = partially supported
    2 = weakly supported
    1 = mostly unsupported
    0 = contradicted or unsupported

    Return ONLY valid JSON with this schema:
    {{
    "correct": 0 or 1,
    "score": integer 0-5,
    "reason": "short explanation",
    "faithful": 0 or 1,
    "faithfulness_score": integer 0-5,
    "faithfulness_reason": "short explanation"
    }}

    Question:
    {question}

    Gold answer:
    {gold_answer}

    Predicted answer:
    {pred_answer}

    Evidence:
    {evidence}
"""

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s

def strip_citations(s: str) -> str:
    s = re.sub(r"\[[^\]]+\]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))

def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)

def retrieval_recall(pred_ids: List[str], gold_ids: List[str]) -> float:
    if not gold_ids:
        return 0.0
    pred = set(pred_ids)
    gold = set(gold_ids)
    return len(pred & gold) / len(gold)

def build_evidence_text(row: Dict) -> str:
    blocks: List[str] = []

    for q in row.get("selected_text_quotes", []):
        blocks.append(f"[{q['quote_id']}] TEXT: {q.get('text', '')}")

    for q in row.get("selected_img_quotes", []):
        desc = q.get("img_description", "")
        blocks.append(f"[{q['quote_id']}] IMAGE: {desc}")

    return "\n".join(blocks)

def llm_judge(
    client: OpenAI,
    judge_model: str,
    question: str,
    gold_answer: str,
    pred_answer: str,
    evidence: str,
) -> Dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        pred_answer=strip_citations(pred_answer),
        evidence=evidence if evidence.strip() else "No evidence provided.",
    )

    resp = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {
            "correct": 0,
            "score": 0,
            "reason": f"Judge returned invalid JSON: {text[:200]}",
            "faithful": 0,
            "faithfulness_score": 0,
            "faithfulness_reason": "Judge returned invalid JSON",
        }

    return {
        "correct": int(parsed.get("correct", 0)),
        "score": int(parsed.get("score", 0)),
        "reason": str(parsed.get("reason", "")),
        "faithful": int(parsed.get("faithful", 0)),
        "faithfulness_score": int(parsed.get("faithfulness_score", 0)),
        "faithfulness_reason": str(parsed.get("faithfulness_reason", "")),
    }

def lexical_metrics(pred: str, gold: str) -> Dict[str, float]:
    pred_clean = strip_citations(pred)
    return {
        "em": exact_match(pred_clean, gold),
        "f1": token_f1(pred_clean, gold),
    }

def aggregate_summary(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "num_examples": 0,
            "avg_retrieval_sec": None,
            "avg_ttft_sec": None,
            "avg_generation_sec": None,
            "avg_total_sec": None,
        }

    def avg_metric(name: str) -> Optional[float]:
        vals = [r["metrics"][name] for r in rows if name in r.get("metrics", {})]
        return (sum(vals) / len(vals)) if vals else None

    valid_ttft = [r["timing"]["ttft_sec"] for r in rows if r["timing"]["ttft_sec"] is not None]

    summary = {
        "num_examples": len(rows),
        "avg_retrieval_sec": sum(r["timing"]["retrieval_sec"] for r in rows) / len(rows),
        "avg_ttft_sec": (sum(valid_ttft) / len(valid_ttft)) if valid_ttft else None,
        "avg_generation_sec": sum(r["timing"]["generation_sec"] for r in rows) / len(rows),
        "avg_total_sec": sum(r["timing"]["total_sec"] for r in rows) / len(rows),
    }

    for metric_name in [
        "em",
        "f1",
        "retrieval_recall",
        "judge_correct",
        "judge_score",
        "judge_faithful",
        "judge_faithfulness_score",
    ]:
        value = avg_metric(metric_name)
        if value is not None:
            summary[f"avg_{metric_name}"] = value

    return summary

