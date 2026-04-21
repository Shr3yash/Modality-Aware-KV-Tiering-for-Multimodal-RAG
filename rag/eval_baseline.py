import json
from pathlib import Path
from statistics import mean

from openai import OpenAI

from rag.config import RAGConfig
from rag.mmdocrag_dataset import load_examples
from rag.metrics import aggregate_summary, build_evidence_text, lexical_metrics, llm_judge, retrieval_recall
from rag.query_pipeline import MMDocRAGPipeline

def _predictions_path(cfg: RAGConfig) -> Path:
    return cfg.output_dir / "baseline_predictions.json"

def _judged_path(cfg: RAGConfig) -> Path:
    return cfg.output_dir / "baseline_results_judged.json"

def run_baseline(cfg: RAGConfig):
    """
    Online serving pass only.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(cfg.ann_file, cfg.images_root, limit=cfg.max_examples)
    pipe = MMDocRAGPipeline(cfg)

    rows = []
    for i, ex in enumerate(examples, start=1):
        out = pipe.run_one(ex)
        out["metrics"] = {
            **lexical_metrics(out["pred_answer"], out["gold_answer"]),
            "retrieval_recall": retrieval_recall(out["retrieved_quote_ids"], out["gold_quotes"]),
        }
        rows.append(out)
        print(
            f"[{i}/{len(examples)}] "
            f"EM={out['metrics']['em']:.2f} "
            f"F1={out['metrics']['f1']:.2f} "
            f"R-Recall={out['metrics']['retrieval_recall']:.2f} "
            f"TTFT={out['timing']['ttft_sec'] if out['timing']['ttft_sec'] is not None else 'NA'} "
            f"Total={out['timing']['total_sec']:.2f}s"
        )
    valid_ttft = [r["timing"]["ttft_sec"] for r in rows if r["timing"]["ttft_sec"] is not None]
    summary = aggregate_summary(rows)
    summary["pruning_mode"] = cfg.pruning_mode
    summary["pruning_keep_ratio"] = cfg.pruning_keep_ratio

    results = {
        "summary": summary,
        "rows": rows,
    }
    
    with _predictions_path(cfg).open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

def run_offline_judge(cfg: RAGConfig, predictions_file: Path | None = None):
    """
    Offline evaluation pass.
    Reads saved predictions and runs LLM-as-a-judge separately from the serving benchmark.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = predictions_file or _predictions_path(cfg)

    with predictions_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data["rows"]
    judge_client = OpenAI(base_url=cfg.judge_api_base, api_key="EMPTY")

    for i, row in enumerate(rows, start=1):
        evidence = build_evidence_text(row)
        judge = llm_judge(
            client=judge_client,
            judge_model=cfg.judge_model_name,
            question=row["question"],
            gold_answer=row["gold_answer"],
            pred_answer=row["pred_answer"],
            evidence=evidence,
        )
        row.setdefault("metrics", {})
        row["metrics"].update({
            "judge_correct": judge["correct"],
            "judge_score": judge["score"],
            "judge_faithful": judge["faithful"],
            "judge_faithfulness_score": judge["faithfulness_score"],
        })
        row["judge"] = judge
        row["evidence_text"] = evidence

        print(
            f"[judge {i}/{len(rows)}] "
            f"Correct={judge['correct']} "
            f"C-Score={judge['score']} "
            f"Faithful={judge['faithful']} "
            f"F-Score={judge['faithfulness_score']}"
        )
    summary = aggregate_summary(rows)
    summary["pruning_mode"] = cfg.pruning_mode
    summary["pruning_keep_ratio"] = cfg.pruning_keep_ratio
    results = {
        "summary": summary,
        "rows": rows,
    }

    with _judged_path(cfg).open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
