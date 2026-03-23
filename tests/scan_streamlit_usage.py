#!/usr/bin/env python3
"""
Enhanced Streamlit usage & callback safety scanner.

Goals
-----
1) Classify Streamlit usage accurately relative to project root.
2) Detect dangerous callbacks passed to parallel routines (EA, WF, trainer):
   - progress_cb pointing to a function that captures `st` (uses st.*)
   - progress_cb pointing to streamlit_progress
3) Detect guard patterns in callback definitions (MainProcess/session guards).
4) Produce a precise log with line numbers and optional code context.

Usage
-----
python tests/scan_streamlit_usage.py --root . --log logs/streamlit_scan.log --show-context 2 --fail-on-high
"""

from __future__ import annotations
import argparse, json, os, re, sys, textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ast

UI_DIR = Path("pages")

# Regex fallbacks (kept from earlier version for light-weight checks)
RE_IMPORT_ST = re.compile(r"^\s*import\s+streamlit\b|^\s*from\s+streamlit\b", re.IGNORECASE)
RE_ST_DOT    = re.compile(r"(^|[^A-Za-z0-9_])st\.[A-Za-z_][A-Za-z0-9_]*")
RE_PAGE_CFG  = re.compile(r"st\.set_page_config\s*\(")
RE_USE_CONT  = re.compile(r"use_container_width\s*=")
RE_SESSION   = re.compile(r"st\.runtime.*exists|is_session_active\s*\(")
RE_MAINPROC  = re.compile(r"current_process\(\)\.name.*MainProcess")

# Calls that likely spin workers or run in parallel
PARALLEL_ENTRYPOINTS = {
    "evolutionary_search",
    "walkforward",
    "walk_forward",
    "train_general_model",
    "train_base_model",
    # add others if needed
}

@dataclass
class Issue:
    severity: str
    type: str
    line: Optional[int] = None
    code: str = ""
    hint: str = ""

@dataclass
class FileReport:
    rel: str
    in_pages: bool
    issues: List[Issue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ""

def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return os.path.relpath(str(path), str(root))

def get_context(lines: List[str], idx: int, n: int) -> str:
    if n <= 0: return ""
    start = max(0, idx - n - 1)
    end = min(len(lines), idx + n)
    snippet = lines[start:end]
    ln = start + 1
    return "\n".join(f"{ln+i:>5}: {s}" for i, s in enumerate(snippet))

def function_uses_streamlit(fn: ast.FunctionDef) -> bool:
    """
    Heuristic: function body references `st.` or imports streamlit.
    """
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "st":
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "streamlit":
                    return True
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("streamlit"):
            return True
    return False

def function_has_guard(fn: ast.FunctionDef) -> bool:
    """
    Heuristic: function contains MainProcess/session guard patterns.
    """
    src = ast.get_source_segment  # may be None but we only use patterns via AST below
    has_mp = False
    has_session = False
    # Check simple textual patterns inside function
    fn_src = ""
    try:
        fn_src = textwrap.dedent(ast.unparse(fn))  # py>=3.9; if unavailable, guards below won’t trigger
    except Exception:
        pass
    if fn_src:
        if "current_process" in fn_src and "MainProcess" in fn_src:
            has_mp = True
        if "st.runtime" in fn_src and "exists" in fn_src:
            has_session = True
    # AST fallbacks
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if getattr(node.func.value, "id", "") == "st" and node.func.attr == "runtime":
                has_session = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "current_process":
                has_mp = True
    return has_mp or has_session

def find_name_binding(name: str, tree: ast.AST) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None

def arg_value_to_str(node: ast.AST, lines: List[str]) -> str:
    # Best-effort code extract for the arg value
    try:
        return ast.unparse(node)  # py>=3.9
    except Exception:
        # fallback: get physical line
        if hasattr(node, "lineno"):
            return lines[getattr(node, "lineno", 1)-1].strip()
        return "<expr>"

def scan_python_file(path: Path, root: Path, show_ctx: int) -> FileReport:
    rel = safe_relpath(path, root)
    in_pages = rel.split(os.sep, 1)[0] == "pages"  # accurate classification
    text = read_text(path)
    lines = text.splitlines()
    report = FileReport(rel=rel, in_pages=in_pages)

    # 1) Lightweight regex scan (general hygiene)
    has_import_st_top = False
    has_any_st_dot = False
    has_page_cfg = False
    use_container_refs = 0
    guards_found = False

    for i, line in enumerate(lines, 1):
        if RE_IMPORT_ST.search(line):
            if not line.startswith((" ", "\t")):
                has_import_st_top = True
            if not in_pages:
                report.issues.append(Issue("HIGH", "import_streamlit_outside_pages", i, line.strip()))
        if RE_ST_DOT.search(line) and not in_pages:
            has_any_st_dot = True
            report.issues.append(Issue("HIGH", "st_attribute_outside_pages", i, line.strip()))
        if RE_PAGE_CFG.search(line):
            has_page_cfg = True
            sev = "LOW" if in_pages else "MEDIUM"
            report.issues.append(Issue(sev, "set_page_config", i, line.strip()))
        if RE_USE_CONT.search(line):
            use_container_refs += 1
            report.issues.append(Issue("LOW", "use_container_width_deprecated", i, line.strip()))
        if RE_MAINPROC.search(line) or RE_SESSION.search(line):
            guards_found = True

    # 2) AST pass for callback analysis
    try:
        tree = ast.parse(text)
    except Exception as e:
        report.issues.append(Issue("HIGH", "parse_error", None, "", f"{e}"))
        report.summary = {
            "has_import_st_top": has_import_st_top,
            "has_any_st_dot": has_any_st_dot,
            "has_page_cfg": has_page_cfg,
            "use_container_refs": use_container_refs,
            "guards_found": guards_found,
        }
        return report

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            # Identify function name
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if not func_name:
                return

            # Only analyze interesting entrypoints
            if func_name not in PARALLEL_ENTRYPOINTS:
                return

            # Try to locate progress_cb
            progress_node = None
            for kw in node.keywords or []:
                if kw.arg == "progress_cb":
                    progress_node = kw.value
                    break

            # Heuristic positional arg discovery (last arg may be a callback)
            if not progress_node and node.args:
                last = node.args[-1]
                # If last arg looks like a name and is a function in this module, treat as candidate
                if isinstance(last, ast.Name):
                    progress_node = last

            if not progress_node:
                return

            # Case 1: Name -> function defined in this file
            if isinstance(progress_node, ast.Name):
                fn_def = find_name_binding(progress_node.id, tree)
                if isinstance(fn_def, ast.FunctionDef):
                    uses_st = function_uses_streamlit(fn_def)
                    has_guard = function_has_guard(fn_def)
                    sev = "HIGH" if uses_st and not has_guard else ("MEDIUM" if uses_st else "LOW")
                    code = f"{func_name}(...) progress_cb={progress_node.id}"
                    hint = "Callback captures Streamlit; add MainProcess/session guard or pass console_progress." if uses_st else ""
                    if has_guard and uses_st:
                        hint += " (Guard detected; severity reduced.)"
                    line_no = getattr(node, "lineno", None)
                    ctx = ""
                    if show_ctx:
                        ctx = get_context(lines, line_no or 1, show_ctx)
                        code += f"\n{ctx}"
                    report.issues.append(Issue(sev, "DANGEROUS_CALLBACK_STREAMLIT_CAPTURE", line_no, code, hint))
                    return

            # Case 2: Attribute or Name that resolves to streamlit_progress or similar
            val_str = arg_value_to_str(progress_node, lines)
            if "streamlit_progress" in val_str:
                line_no = getattr(node, "lineno", None)
                ctx = get_context(lines, line_no or 1, show_ctx) if show_ctx else ""
                report.issues.append(Issue(
                    "HIGH",
                    "DANGEROUS_CALLBACK_STREAMLIT_PROGRESS",
                    line_no,
                    f"{func_name}(...) progress_cb={val_str}\n{ctx}",
                    "Do not pass streamlit_progress into worker code; use console_progress."
                ))
                return

            # Case 3: Lambda referencing st.*
            if isinstance(progress_node, ast.Lambda):
                uses_st = any(
                    isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "st"
                    for n in ast.walk(progress_node)
                )
                if uses_st:
                    line_no = getattr(node, "lineno", None)
                    ctx = get_context(lines, line_no or 1, show_ctx) if show_ctx else ""
                    report.issues.append(Issue(
                        "HIGH",
                        "DANGEROUS_CALLBACK_LAMBDA_STREAMLIT",
                        line_no,
                        f"{func_name}(...) progress_cb=<lambda>\n{ctx}",
                        "Lambda captures Streamlit; use console_progress."
                    ))
                    return

    CallVisitor().visit(tree)

    # 3) Meta issue: Streamlit used under src/ without guards
    if (has_import_st_top or has_any_st_dot) and not in_pages and not guards_found:
        report.issues.append(Issue(
            "HIGH", "no_guard_detected_for_streamlit_usage", None, "",
            "Add MainProcess/session guard or move Streamlit code to UI layer."
        ))

    report.summary = {
        "has_import_st_top": has_import_st_top,
        "has_any_st_dot": has_any_st_dot,
        "has_page_cfg": has_page_cfg,
        "use_container_refs": use_container_refs,
        "guards_found": guards_found,
    }
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root (default: .)")
    ap.add_argument("--log", default="logs/streamlit_scan.log", help="Output log file")
    ap.add_argument("--show-context", type=int, default=0, help="Show N lines of context around hits")
    ap.add_argument("--fail-on-high", action="store_true", help="Exit 1 if HIGH issues are found")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.log)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect Python files under src/ and pages/
    files: List[Path] = []
    for sub in ("src", "pages"):
        p = root / sub
        if p.exists():
            files.extend(sorted(q for q in p.rglob("*.py") if q.is_file()))

    reports: List[FileReport] = []
    for f in files:
        reports.append(scan_python_file(f, root, args.show_context))

    # Summaries
    high = sum(len([i for i in r.issues if i.severity == "HIGH"]) for r in reports)
    med  = sum(len([i for i in r.issues if i.severity == "MEDIUM"]) for r in reports)
    low  = sum(len([i for i in r.issues if i.severity == "LOW"]) for r in reports)

    # Human-readable log
    out_lines: List[str] = []
    out_lines.append("# Streamlit usage scan (enhanced)")
    out_lines.append(f"root: {root}")
    out_lines.append(f"files_scanned: {len(reports)}\n")
    out_lines.append(f"Totals → HIGH: {high} | MEDIUM: {med} | LOW: {low}\n")

    for r in reports:
        if not r.issues:
            continue
        out_lines.append(f"== {r.rel} ==")
        for iss in r.issues:
            loc = f" line {iss.line}" if iss.line else ""
            hint = f" | hint: {iss.hint}" if iss.hint else ""
            code = f"\n{iss.code}" if iss.code else ""
            out_lines.append(f"  [{iss.severity}] {iss.type}{loc}{hint}{code}")
        out_lines.append("")

    # JSONL appendix
    out_lines.append("# JSONL")
    for r in reports:
        obj = {
            "file": r.rel,
            "in_pages": r.in_pages,
            "issues": [iss.__dict__ for iss in r.issues],
            "summary": r.summary
        }
        out_lines.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote scan → {out_path}")

    if args.fail_on_high and high > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()