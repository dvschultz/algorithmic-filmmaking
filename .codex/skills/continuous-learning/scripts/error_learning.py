#!/usr/bin/env python3
"""
Record recurring errors and promote stable fixes into reusable playbook skills.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ParsedError:
    signature: str
    title: str
    exception_type: str
    exception_message: str
    primary_frame: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, max_len: int = 50) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    value = re.sub(r"-{2,}", "-", value)
    return value[:max_len].strip("-") or "error-pattern"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_memory_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at": utc_now_iso(),
                "patterns": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_memory(path: Path) -> dict:
    ensure_memory_file(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "patterns" not in data or not isinstance(data["patterns"], list):
        raise ValueError(f"Invalid memory schema at {path}")
    return data


def save_memory(path: Path, data: dict) -> None:
    data["updated_at"] = utc_now_iso()
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_traceback(error_text: str) -> ParsedError:
    lines = [line.rstrip() for line in error_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("No error text provided")

    frame_matches = re.findall(
        r'File "([^"]+)", line (\d+), in ([^\n]+)', error_text
    )

    primary_frame = "unknown-frame"
    for file_path, line_no, fn_name in frame_matches:
        if "/algorithmic-filmmaking/" in file_path or file_path.startswith(("ui/", "core/", "models/", "cli/")):
            primary_frame = f"{Path(file_path).as_posix()}:{line_no}:{fn_name.strip()}"
            break
    if primary_frame == "unknown-frame" and frame_matches:
        fp, ln, fn = frame_matches[-1]
        primary_frame = f"{Path(fp).as_posix()}:{ln}:{fn.strip()}"

    exception_line = lines[-1]
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.+)$", exception_line)
    if m:
        exc_type = m.group(1)
        exc_msg = m.group(2).strip()
    else:
        exc_type = slugify(exception_line, max_len=30).replace("-", "_")
        exc_msg = exception_line

    title = f"{exc_type}: {exc_msg}"[:120]
    signature = f"{exc_type}|{primary_frame}"

    return ParsedError(
        signature=signature,
        title=title,
        exception_type=exc_type,
        exception_message=exc_msg,
        primary_frame=primary_frame,
    )


def parse_frontmatter(md_text: str) -> dict:
    if not md_text.startswith("---\n"):
        return {}
    end = md_text.find("\n---\n", 4)
    if end == -1:
        return {}
    raw = md_text[4:end]
    out = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def build_signature_from_solution_doc(frontmatter: dict, rel_path: Path) -> str:
    component = frontmatter.get("component", "unknown-component")
    root_cause = frontmatter.get("root_cause", "unknown-cause")
    symptom = frontmatter.get("symptom", "")
    if not symptom and "symptoms" in frontmatter:
        symptom = frontmatter["symptoms"]
    if not symptom:
        symptom = rel_path.stem
    return f"{slugify(component)}|{slugify(root_cause)}|{slugify(str(symptom), max_len=40)}"


def short_description_for_playbook(title: str) -> str:
    text = f"Fix recurring: {title}"
    if len(text) > 64:
        text = text[:64].rstrip()
    if len(text) < 25:
        text = f"{text} playbook"
    return text


def promote_to_playbook_skill(
    repo_root: Path,
    target_dir: Path,
    pattern: dict,
) -> Path:
    signature_slug = slugify(pattern["signature"], max_len=40)
    skill_name = f"error-playbook-{signature_slug}"
    skill_name = skill_name[:64].rstrip("-")
    skill_dir = target_dir / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)
    (skill_dir / "agents").mkdir(parents=True, exist_ok=True)

    title = pattern.get("title", pattern["signature"])
    refs = pattern.get("solution_refs", [])
    refs_md = "\n".join(f"- `{r}`" for r in refs) if refs else "- (none yet)"

    skill_md = textwrap.dedent(
        f"""\
        ---
        name: {skill_name}
        description: Resolve recurring error pattern '{pattern["signature"]}' for Scene Ripper workflows. Use when logs or tracebacks match this signature or symptom family.
        ---

        # {title}

        ## Workflow

        1. Confirm the traceback/error output matches signature `{pattern["signature"]}`.
        2. Validate the failing component before changing unrelated modules.
        3. Apply the documented fix pattern from the references.
        4. Add or update regression tests and verify the failure no longer reproduces.

        ## References

        {refs_md}
        """
    )
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    openai_yaml = textwrap.dedent(
        f"""\
        interface:
          display_name: "{title[:64]}"
          short_description: "{short_description_for_playbook(title)}"
          default_prompt: "Diagnose this traceback, confirm it matches the playbook, and apply the documented fix safely."
        """
    )
    (skill_dir / "agents" / "openai.yaml").write_text(openai_yaml, encoding="utf-8")

    notes_md = textwrap.dedent(
        f"""\
        # Notes

        - Signature: `{pattern["signature"]}`
        - First seen: `{pattern.get("first_seen")}`
        - Last seen: `{pattern.get("last_seen")}`
        - Occurrences: `{pattern.get("occurrences", 0)}`

        ## Solution References

        {refs_md}
        """
    )
    (skill_dir / "references" / "notes.md").write_text(notes_md, encoding="utf-8")
    try:
        return skill_dir.relative_to(repo_root)
    except ValueError:
        return skill_dir


def find_or_create_pattern(data: dict, signature: str, title: str, problem_type: str) -> dict:
    for pattern in data["patterns"]:
        if pattern.get("signature") == signature:
            return pattern

    created = {
        "signature": signature,
        "title": title,
        "problem_type": problem_type,
        "occurrences": 0,
        "first_seen": utc_now_iso(),
        "last_seen": utc_now_iso(),
        "solution_refs": [],
        "symptom_examples": [],
        "playbook_skill": None,
    }
    data["patterns"].append(created)
    return created


def command_record(args: argparse.Namespace) -> int:
    memory_path = Path(args.memory).resolve()
    repo_root = Path(args.repo_root).resolve()
    target_dir = (repo_root / args.promote_dir).resolve()

    data = load_memory(memory_path)

    if args.signature:
        parsed = ParsedError(
            signature=args.signature.strip(),
            title=args.title or args.signature.strip(),
            exception_type="custom",
            exception_message=args.title or args.signature.strip(),
            primary_frame="custom",
        )
    else:
        error_text = args.error_text or ""
        if args.error_file:
            error_text = read_text(Path(args.error_file))
        parsed = parse_traceback(error_text)
        if args.title:
            parsed.title = args.title

    pattern = find_or_create_pattern(
        data=data,
        signature=parsed.signature,
        title=parsed.title,
        problem_type=args.problem_type,
    )

    pattern["occurrences"] = int(pattern.get("occurrences", 0)) + 1
    pattern["last_seen"] = utc_now_iso()
    pattern["title"] = parsed.title

    if args.symptom and args.symptom not in pattern["symptom_examples"]:
        pattern["symptom_examples"].append(args.symptom)
    elif not args.symptom and parsed.exception_message:
        msg = parsed.exception_message[:160]
        if msg and msg not in pattern["symptom_examples"]:
            pattern["symptom_examples"].append(msg)

    for ref in args.solution_ref:
        ref = str(Path(ref).as_posix())
        if ref not in pattern["solution_refs"]:
            pattern["solution_refs"].append(ref)

    promoted = False
    if (
        pattern["occurrences"] >= args.promote_threshold
        and pattern["solution_refs"]
        and not pattern.get("playbook_skill")
    ):
        rel_skill_path = promote_to_playbook_skill(
            repo_root=repo_root,
            target_dir=target_dir,
            pattern=pattern,
        )
        pattern["playbook_skill"] = str(rel_skill_path.as_posix())
        promoted = True

    save_memory(memory_path, data)

    print(f"[OK] signature: {pattern['signature']}")
    print(f"[OK] occurrences: {pattern['occurrences']}")
    if pattern["solution_refs"]:
        print(f"[OK] refs: {len(pattern['solution_refs'])}")
    if promoted:
        print(f"[OK] promoted to skill: {pattern['playbook_skill']}")
    return 0


def command_bootstrap(args: argparse.Namespace) -> int:
    memory_path = Path(args.memory).resolve()
    repo_root = Path(args.repo_root).resolve()
    solutions_root = (repo_root / args.solutions_dir).resolve()
    if not solutions_root.exists():
        raise FileNotFoundError(f"Solutions directory not found: {solutions_root}")

    data = load_memory(memory_path)
    imported = 0

    for md_path in sorted(solutions_root.rglob("*.md")):
        rel = md_path.relative_to(repo_root)
        frontmatter = parse_frontmatter(read_text(md_path))
        signature = build_signature_from_solution_doc(frontmatter, rel)
        title = frontmatter.get("title", rel.stem.replace("-", " "))
        ptype = frontmatter.get("problem_type") or frontmatter.get("category") or "incident"
        pattern = find_or_create_pattern(data, signature, title, ptype)

        if rel.as_posix() not in pattern["solution_refs"]:
            pattern["solution_refs"].append(rel.as_posix())
        if pattern["occurrences"] == 0:
            pattern["occurrences"] = 1
        pattern["last_seen"] = utc_now_iso()
        imported += 1

    save_memory(memory_path, data)
    print(f"[OK] bootstrapped {imported} solution docs into {memory_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuous error-learning utility for Scene Ripper."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--memory",
        default="docs/learning/error_memory.json",
        help="Path to error memory JSON file",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Record a new error occurrence")
    record.add_argument("--signature", help="Explicit error signature")
    record.add_argument("--title", help="Optional human title")
    record.add_argument("--error-file", help="Path to traceback/error text file")
    record.add_argument("--error-text", help="Raw traceback text")
    record.add_argument("--problem-type", default="runtime_error")
    record.add_argument("--symptom", help="Short symptom example text")
    record.add_argument(
        "--solution-ref",
        action="append",
        default=[],
        help="Path to fix documentation (repeatable)",
    )
    record.add_argument(
        "--promote-threshold",
        type=int,
        default=3,
        help="Occurrences required before generating a playbook skill",
    )
    record.add_argument(
        "--promote-dir",
        default="skills/error-playbooks",
        help="Directory where generated playbook skills are created",
    )

    bootstrap = sub.add_parser(
        "bootstrap",
        help="Seed error memory from docs/solutions markdown files",
    )
    bootstrap.add_argument(
        "--solutions-dir",
        default="docs/solutions",
        help="Directory containing solution markdown files",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "record":
        return command_record(args)
    if args.command == "bootstrap":
        return command_bootstrap(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
