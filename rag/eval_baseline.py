import json
from pathlib import Path

from openai import OpenAI

from rag.config import RAGConfig
from rag.mmdocrag_dataset import load_examples
from rag.metrics import aggregate_summary, lexical_metrics, llm_judge, retrieval_recall
from rag.query_pipeline import RAGPipeline
from rag.vqa_metrics import vqa_accuracy
from rag.vqav2_dataset import load_vqav2_examples

def _predictions_path(cfg: RAGConfig) -> Path:
    return cfg.output_dir / "baseline_predictions.json"

def _judged_path(cfg: RAGConfig) -> Path:
    return cfg.output_dir / "baseline_results_judged.json"

def run_baseline(cfg: RAGConfig):
    """
    Online serving pass only.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.benchmark == "vqav2":
        examples = load_vqav2_examples(cfg)
    elif cfg.benchmark == "mmdocrag":
        examples = load_examples(
            cfg.ann_file,
            cfg.images_root,
            limit=cfg.max_examples,
            slice_start=cfg.eval_slice_start,
            slice_stop=cfg.eval_slice_stop,
        )
    else:
        raise ValueError(f"Unknown benchmark: {cfg.benchmark!r}")

    pipe = RAGPipeline(cfg)

    rows = []
    for i, ex in enumerate(examples, start=1):
        out = pipe.run_one(ex)
        out["vqa_answers"] = ex.get("vqa_answers", [])
        metrics = {
            **lexical_metrics(out["pred_answer"], out["gold_answer"]),
            "retrieval_recall": retrieval_recall(out["retrieved_quote_ids"], out["gold_quotes"]),
        }
        if ex.get("vqa_answers"):
            metrics["vqa_acc"] = vqa_accuracy(out["pred_answer"], ex["vqa_answers"])
        out["metrics"] = metrics
        rows.append(out)
        extra = ""
        if "vqa_acc" in metrics:
            extra = f" VQA_acc={metrics['vqa_acc']:.2f}"
        print(
            f"[{i}/{len(examples)}] "
            f"EM={out['metrics']['em']:.2f} "
            f"F1={out['metrics']['f1']:.2f} "
            f"R-Recall={out['metrics']['retrieval_recall']:.2f}"
            f"{extra} "
            f"TTFT={out['timing']['ttft_sec'] if out['timing']['ttft_sec'] is not None else 'NA'} "
            f"Total={out['timing']['total_sec']:.2f}s"
        )
    valid_ttft = [r["timing"]["ttft_sec"] for r in rows if r["timing"]["ttft_sec"] is not None]
    summary = aggregate_summary(rows)
    summary["benchmark"] = cfg.benchmark
    summary["pruning_mode"] = cfg.pruning_mode
    summary["pruning_keep_ratio"] = cfg.pruning_keep_ratio

    results = {
        "summary": summary,
        "rows": rows,
        "benchmark": cfg.benchmark,
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
    benchmark = data.get("summary", {}).get("benchmark") or data.get("benchmark") or cfg.benchmark
    judge_client = OpenAI(base_url=cfg.judge_api_base, api_key="EMPTY")

    for i, row in enumerate(rows, start=1):
        evidence = ""
        judge = llm_judge(
            client=judge_client,
            judge_model=cfg.judge_model_name,
            question=row["question"],
            gold_answer=row["gold_answer"],
            pred_answer=row["pred_answer"],
        )
        row.setdefault("metrics", {})
        row["metrics"].update({
            "judge_correct": judge["correct"],
            "judge_score": judge["score"],
        })
        row["judge"] = judge

        print(
            f"[judge {i}/{len(rows)}] "
            f"Correct={judge['correct']} "
            f"C-Score={judge['score']} "
        )
    summary = aggregate_summary(rows)
    summary["benchmark"] = benchmark
    summary["pruning_mode"] = cfg.pruning_mode
    summary["pruning_keep_ratio"] = cfg.pruning_keep_ratio

    results = {
        "summary": summary,
        "rows": rows,
        "benchmark": benchmark,
    }

    with _judged_path(cfg).open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
