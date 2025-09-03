#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu, sentence_bleu


def _load_pairs_from_csv(refs_csv: Path) -> Tuple[List[str], List[str]]:
    """Load reference (gold) answers matched by question text."""
    df = pd.read_csv(refs_csv)
    # expects columns: id,question,answer
    if not {"question", "answer"}.issubset(df.columns):
        raise SystemExit("qa_short_answers.csv must have columns: question, answer")
    questions = df["question"].astype(str).tolist()
    references = df["answer"].astype(str).tolist()
    return questions, references


def _load_questions(questions_csv: Path) -> List[str]:
    df = pd.read_csv(questions_csv)
    # expects columns: category,type,question,wikipedia_url (url may be empty)
    if "question" not in df.columns:
        raise SystemExit("questions.csv must have a 'question' column")
    return df["question"].astype(str).tolist()


def _ensure_outdir(path: Path) -> Path:
    outdir = path if path is not None else Path("out")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _avg(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


@dataclass
class RougeTriple:
    f1: float
    precision: float
    recall: float


def _eval_text(references: List[str], hypotheses: List[str]) -> pd.DataFrame:
    if len(references) != len(hypotheses) or len(references) == 0:
        raise SystemExit("Need equal, non-zero counts of references and predictions.")

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True
    )

    rows = []
    for ref, hyp in zip(references, hypotheses):
        sbleu = sentence_bleu(hyp, [ref]).score
        rs = scorer.score(ref, hyp)
        r1, r2, rL = rs["rouge1"], rs["rouge2"], rs["rougeLsum"]
        rows.append(
            {
                "reference": ref,
                "prediction": hyp,
                "bleu": sbleu,
                "rouge1_f1": r1.fmeasure,
                "rouge1_p": r1.precision,
                "rouge1_r": r1.recall,
                "rouge2_f1": r2.fmeasure,
                "rouge2_p": r2.precision,
                "rouge2_r": r2.recall,
                "rougeLsum_f1": rL.fmeasure,
                "rougeLsum_p": rL.precision,
                "rougeLsum_r": rL.recall,
            }
        )
    return pd.DataFrame(rows)


def _ask_openai(
    questions: List[str], model: str, max_tokens: int, temperature: float
) -> List[str]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    system = (
        "Answer in plain English, ≤35 words. No preamble, no lists, no formatting, no links. "
        "Be direct and factual."
    )

    answers: List[str] = []
    for q in questions:
        # Using chat.completions for widest compatibility
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": q},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answers.append(resp.choices[0].message.content.strip())
    return answers


def main():
    p = argparse.ArgumentParser(
        description="Ask OpenAI for short answers, then evaluate with BLEU/ROUGE and write JSON report."
    )
    p.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="questions.csv (with column 'question').",
    )
    p.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="qa_short_answers.csv (columns: question, answer).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("out"),
        help="Output directory (default: out).",
    )
    p.add_argument(
        "--model", default="gpt-4.1-mini", help="OpenAI model (default: gpt-4.1-mini)."
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Max tokens per answer (default: 80).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    args = p.parse_args()

    outdir = _ensure_outdir(args.outdir)

    # Load data
    questions_all = _load_questions(args.questions)
    gold_qs, gold_refs = _load_pairs_from_csv(args.gold)

    # Align: keep only questions that exist in gold file, preserving input order
    q_set = set(gold_qs)
    questions = [q for q in questions_all if q in q_set]
    if not questions:
        raise SystemExit(
            "No overlapping questions between questions.csv and qa_short_answers.csv"
        )

    # Map question -> reference
    ref_map = {q: r for q, r in zip(gold_qs, gold_refs)}
    references = [ref_map[q] for q in questions]

    # Ask OpenAI
    predictions = _ask_openai(questions, args.model, args.max_tokens, args.temperature)

    # Save raw OpenAI answers
    answers_path = outdir / "openai_answers.json"
    with answers_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {"question": q, "openai_answer": a}
                for q, a in zip(questions, predictions)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Evaluate
    per_df = _eval_text(references, predictions)

    # Build unified JSON report: include question, reference, prediction, metrics columns requested
    report_items = []
    for q, ref, pred, row in zip(
        questions, references, predictions, per_df.to_dict(orient="records")
    ):
        item = {
            "question": q,
            "reference": ref,
            "prediction": pred,
            "bleu": row["bleu"],
            "rouge1_f1": row["rouge1_f1"],
            "rouge1_p": row["rouge1_p"],
            "rouge1_r": row["rouge1_r"],
            "rouge2_f1": row["rouge2_f1"],
            "rouge2_p": row["rouge2_p"],
            "rouge2_r": row["rouge2_r"],
            "rougeLsum_f1": row["rougeLsum_f1"],
            "rougeLsum_p": row["rougeLsum_p"],
            "rougeLsum_r": row["rougeLsum_r"],
        }
        report_items.append(item)

    report_path = outdir / "report.metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_items, f, ensure_ascii=False, indent=2)

    print(f"Wrote OpenAI answers: {answers_path}")
    print(f"Wrote metrics report: {report_path}")


if __name__ == "__main__":
    main()
