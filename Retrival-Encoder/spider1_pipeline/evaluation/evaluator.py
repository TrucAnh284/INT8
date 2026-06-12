"""
Spider 1.0 Evaluation: Execution Accuracy (EX) + Exact Set Match (EM).

EX: run both gold and predicted SQL on the database; compare result sets.
EM: token-level comparison ignoring value literals (Spider original metric).
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    question_id: int
    db_id: str
    question: str
    gold_sql: str
    pred_sql: str
    exec_match: Optional[bool] = None   # None = execution error
    exact_match: Optional[bool] = None
    error: str = ""


@dataclass
class EvaluationReport:
    total: int = 0
    exec_correct: int = 0
    exact_correct: int = 0
    exec_errors: int = 0
    results: list[ExecutionResult] = field(default_factory=list)

    @property
    def execution_accuracy(self) -> float:
        denom = self.total - self.exec_errors
        return self.exec_correct / denom if denom > 0 else 0.0

    @property
    def exact_match_accuracy(self) -> float:
        return self.exact_correct / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Total questions  : {self.total}",
            f"Exec errors      : {self.exec_errors}",
            f"Exec Accuracy    : {self.execution_accuracy:.4f}  "
            f"({self.exec_correct}/{self.total - self.exec_errors})",
            f"Exact Match (EM) : {self.exact_match_accuracy:.4f}  "
            f"({self.exact_correct}/{self.total})",
        ]
        return "\n".join(lines)


# ── SQL execution helpers ─────────────────────────────────────────────────────

def _adjust_sql(sql: str) -> str:
    """Replace CURDATE() references for offline evaluation."""
    return re.sub(
        r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)", "2020", sql, flags=re.IGNORECASE
    )


def _rows_canonical(raw_rows) -> frozenset:
    """
    Convert query result rows to a canonical, column-order-insensitive form.

    Each row is represented as a sorted tuple of stringified values so that
    ``SELECT a, b`` and ``SELECT b, a`` returning the same data compare equal.
    String values are lower-cased to handle case differences in literals.
    This matches the denotation-equivalence approach used in DAIL-SQL and the
    standard Spider execution-accuracy metric.
    """
    canonical = []
    for row in raw_rows:
        normed = tuple(sorted(str(v).strip().lower() if v is not None else "null" for v in row))
        canonical.append(normed)
    return frozenset(canonical)


def _exec_sql(sql: str, db_path: Path) -> tuple[bool, Optional[frozenset], str]:
    """
    Execute SQL on the SQLite database.
    Returns (success, canonical_result_frozenset, error_message).
    """
    try:
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(_adjust_sql(sql))
        rows = _rows_canonical(cur.fetchall())
        conn.close()
        return True, rows, ""
    except Exception as e:
        return False, None, str(e)


# ── Exact Set Match (simplified) ──────────────────────────────────────────────

_EM_STRIP = re.compile(r"'[^']*'|\"[^\"]*\"|\b\d+(\.\d+)?\b", re.IGNORECASE)


def _normalise_sql(sql: str) -> str:
    """Strip values and normalise whitespace for EM comparison."""
    sql = _EM_STRIP.sub("VALUE", sql)
    sql = re.sub(r"\s+", " ", sql).strip().upper()
    return sql


def exact_match(gold: str, pred: str) -> bool:
    return _normalise_sql(gold) == _normalise_sql(pred)


# ── Main evaluator ────────────────────────────────────────────────────────────

def evaluate_predictions(
    dev_json: Path,
    pred_sql_file: Path,
    database_dir: Path,
    verbose: bool = False,
) -> EvaluationReport:
    """
    Evaluate predicted SQL against the Spider dev set.

    Parameters
    ----------
    dev_json       : path to Spider dev.json (or test.json)
    pred_sql_file  : path to predictions file, one SQL per line
    database_dir   : path to Spider database/ directory
    verbose        : print per-question results

    Returns EvaluationReport.
    """
    questions = json.loads(dev_json.read_text(encoding="utf-8"))
    preds     = [line.strip() for line in pred_sql_file.read_text(encoding="utf-8").splitlines()
                 if line.strip()]

    if len(preds) > len(questions):
        raise ValueError(
            f"Prediction count ({len(preds)}) > question count ({len(questions)})"
        )
    if len(preds) < len(questions):
        print(f"[evaluate] Partial run: evaluating first {len(preds)} of {len(questions)} questions")
        questions = questions[:len(preds)]

    report = EvaluationReport(total=len(questions))

    for i, (item, pred) in enumerate(zip(questions, preds)):
        db_id    = item["db_id"]
        question = item["question"]
        gold     = item["query"]
        db_path  = database_dir / db_id / f"{db_id}.sqlite"

        result = ExecutionResult(
            question_id=i,
            db_id=db_id,
            question=question,
            gold_sql=gold,
            pred_sql=pred,
        )

        # Execution accuracy
        gold_ok, gold_rows, gold_err = _exec_sql(gold, db_path)
        pred_ok, pred_rows, pred_err = _exec_sql(pred, db_path)

        if not gold_ok or not pred_ok:
            result.exec_match = False
            result.error = gold_err or pred_err
            report.exec_errors += 1
        else:
            result.exec_match = (gold_rows == pred_rows)
            if result.exec_match:
                report.exec_correct += 1

        # Exact match
        result.exact_match = exact_match(gold, pred)
        if result.exact_match:
            report.exact_correct += 1

        report.results.append(result)

        if verbose:
            status = "✓" if result.exec_match else "✗"
            print(f"[{i:4d}] {status} {db_id}: {question[:60]}")
            if not result.exec_match and result.error:
                print(f"       Error: {result.error[:80]}")

    return report


def save_report(report: EvaluationReport, output_path: Path) -> None:
    """Serialize report to JSON for analysis."""
    data = {
        "summary": {
            "total":              report.total,
            "exec_errors":        report.exec_errors,
            "exec_accuracy":      report.execution_accuracy,
            "exact_match":        report.exact_match_accuracy,
            "exec_correct":       report.exec_correct,
            "exact_correct":      report.exact_correct,
        },
        "results": [
            {
                "id":          r.question_id,
                "db_id":       r.db_id,
                "question":    r.question,
                "gold_sql":    r.gold_sql,
                "pred_sql":    r.pred_sql,
                "exec_match":  r.exec_match,
                "exact_match": r.exact_match,
                "error":       r.error,
            }
            for r in report.results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
