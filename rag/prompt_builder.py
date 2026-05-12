from typing import Dict, List, Sequence, Union
import json
from pathlib import Path

CachedImg = Union[Sequence[Dict], Dict, None]


def _cached_quotes_list(cached_img: CachedImg) -> List[Dict]:
    if cached_img is None:
        return []
    if isinstance(cached_img, list):
        return list(cached_img)
    return []

def load_cluster_annotations(run_dir: str | Path) -> list[tuple[str, str]]:
    """
    Returns [(cluster_id, annotation), ...] sorted by cluster_id when numeric.
    Assumes files are <cluster_id>.json with an 'annotation' field.
    """
    d = Path(run_dir)
    if not d.is_dir():
        return []

    items: list[tuple[str, str]] = []
    for p in d.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cid = str(data.get("cluster_id", p.stem))
        ann = str(data.get("annotation", "")).strip()
        if ann:
            items.append((cid, ann))

    def key(t: tuple[str, str]):
        cid, _ = t
        try:
            return (0, int(cid))
        except ValueError:
            return (1, cid)

    return sorted(items, key=key)


def _is_single_image_vqa(retrieval: Dict, cached_img: CachedImg) -> bool:
    texts = retrieval.get("selected_text_quotes") or []
    imgs = retrieval.get("selected_img_quotes") or []
    cached = _cached_quotes_list(cached_img)
    return not texts and len(imgs) + len(cached) == 1


def build_prompt(query: str, retrieval: Dict, cached_img: CachedImg) -> str:
    blocks: List[str] = []
    vqa_mode = _is_single_image_vqa(retrieval, cached_img)

    if vqa_mode:
        blocks.append("Answer the visual question using the image provided below.")
        blocks.append("Reply with a short, direct answer (phrase or few words) when possible.")
    else:
        blocks.append("Answer the question using only the retrieved evidence.")
        blocks.append("If the evidence is insufficient, say that clearly.")
        blocks.append("Cite quote IDs you used in brackets, e.g. [text3] [image2].")

    blocks.append("")
    blocks.append(f"Question: {query}")
    blocks.append("")

    if retrieval.get("selected_text_quotes"):
        blocks.append("Retrieved text evidence:")
        for q in retrieval["selected_text_quotes"]:
            blocks.append(f"- [{q['quote_id']}]: {q.get('text', '')}")
        blocks.append("")

    imgs = retrieval.get("selected_img_quotes") or []
    cached = _cached_quotes_list(cached_img)
    if imgs or cached:
        if vqa_mode:
            blocks.append("Image:")
        else:
            blocks.append("Retrieved image evidence:")
        for q in imgs:
            blocks.append(f"- [{q['quote_id']}]")
            run_dir = q.get("local_img_path")
            if run_dir:
                anns = load_cluster_annotations(run_dir)
                if anns:
                    blocks.append("  - clusters:")
                    for cluster_id, annotation in anns:
                        blocks.append(f"    - (cluster_id={cluster_id}) {annotation}")
        for q in cached:
            blocks.append(f"- [{q['quote_id']}]")
            run_dir = q.get("local_img_path")
            if run_dir:
                anns = load_cluster_annotations(run_dir)
                if anns:
                    blocks.append("  - clusters:")
                    for cluster_id, annotation in anns:
                        blocks.append(f"    - (cluster_id={cluster_id}) {annotation}")
        blocks.append("")

    blocks.append("Return a short answer.")
    return "\n".join(blocks)
