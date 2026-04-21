import json
from pathlib import Path
from typing import Dict, List, Iterator

def load_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def resolve_img_path(images_root: Path, img_path: str) -> Path:
    # img_path from dataset is relative; join to local extracted images dir
    return images_root / img_path

def normalize_example(ex: Dict, images_root: Path) -> Dict:
    text_quotes = ex.get("text_quotes", [])
    img_quotes = ex.get("img_quotes", [])

    for q in img_quotes:
        if "img_path" in q:
            q["local_img_path"] = str(resolve_img_path(images_root, q["img_path"]))

    return {
        "q_id": ex["q_id"],
        "doc_name": ex["doc_name"],
        "domain": ex.get("domain"),
        "question": ex["question"],
        "question_type": ex.get("question_type"),
        "evidence_modality_type": ex.get("evidence_modality_type", []),
        "text_quotes": text_quotes,
        "img_quotes": img_quotes,
        "gold_quotes": ex.get("gold_quotes", []),
        "answer_short": ex.get("answer_short", ""),
    }

def load_examples(ann_file: Path, images_root: Path, limit: int | None = None) -> List[Dict]:
    out = []
    for ex in load_jsonl(ann_file):
        out.append(normalize_example(ex, images_root))
        if limit is not None and len(out) >= limit:
            break
    return out