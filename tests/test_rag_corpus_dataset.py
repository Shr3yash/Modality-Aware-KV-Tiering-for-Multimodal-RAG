from pathlib import Path

from datasets import Dataset

from src.rag.corpus import load_corpus


def test_corpus_loader_reads_saved_hf_dataset_structure(tmp_path: Path) -> None:
    dataset = Dataset.from_dict(
        {
            "doc_id": ["doc-a"],
            "markdown": ["cache text image context"],
            "page_number_in_doc": [3],
        }
    )
    dataset.save_to_disk(str(tmp_path))

    chunks = load_corpus(corpus_dir=tmp_path, chunk_size_tokens=2)

    assert chunks
    assert chunks[0].source == "doc-a#page=3"
    assert chunks[0].doc_id == "doc-a"
    assert chunks[0].page_number == 3
    assert chunks[0].text
