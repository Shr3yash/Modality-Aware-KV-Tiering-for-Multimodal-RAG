from typing import Dict, List

def build_prompt(example: Dict, retrieval: Dict) -> str:
    blocks = []
    blocks.append("Answer the question using only the retrieved evidence.")
    blocks.append("If the evidence is insufficient, say that clearly.")
    blocks.append("Cite quote IDs you used in brackets, e.g. [text3] [image2].")
    blocks.append("")
    blocks.append(f"Question: {example['question']}")
    blocks.append("")

    if retrieval["selected_text_quotes"]:
        blocks.append("Retrieved text evidence:")
        for q in retrieval["selected_text_quotes"]:
            blocks.append(
                f"- [{q['quote_id']}] (page {q.get('page_id')}, layout {q.get('layout_id')}): {q.get('text','')}"
            )
        blocks.append("")

    if retrieval["selected_img_quotes"]:
        blocks.append("Retrieved image evidence:")
        for q in retrieval["selected_img_quotes"]:
            blocks.append(
                f"- [{q['quote_id']}] (page {q.get('page_id')}, layout {q.get('layout_id')}): {q.get('img_description','')}"
            )
        blocks.append("")

    blocks.append("Return a short answer.")
    return "\n".join(blocks)