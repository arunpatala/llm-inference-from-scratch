#!/usr/bin/env python3
"""Flag likely AI-writing tells in a markdown chapter.

Banned-word list lives in .claude/skills/write-chapter/references/style_rules.md
under the "## Banned Words" heading — this script parses that file directly so
the list and the checker never drift apart. Everything else here is a soft
heuristic; use judgment, don't chase the exit code to zero by force-fitting text.
"""
import argparse
import re
import sys
from pathlib import Path

RULES_PATH = (
    Path(__file__).resolve().parent.parent
    / ".claude" / "skills" / "write-chapter" / "references" / "style_rules.md"
)

BULLET_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")
HEADING_RE = re.compile(r"^#{1,6}\s+")
BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*")
EM_DASH_RE = re.compile(r"—|--")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]+$")
NEGATIVE_PARALLELISM_RE = re.compile(
    r"\bnot (?:just|only)\b.{0,60}\b(?:but|it'?s)\b", re.IGNORECASE
)


def load_banned_words(rules_path):
    text = rules_path.read_text()
    match = re.search(r"## Banned Words\n(.*?)\n## ", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find '## Banned Words' section in {rules_path}")
    words = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        phrase = line[2:].split(" (say:", 1)[0].strip()
        if phrase:
            words.append(phrase)
    return words


def find_word_hits(lines, banned_words):
    hits = {}
    for phrase in banned_words:
        suffix = r"\w*\b" if " " not in phrase else r"\b"
        pattern = re.compile(r"\b" + re.escape(phrase) + suffix, re.IGNORECASE)
        line_numbers = [i + 1 for i, line in enumerate(lines) if pattern.search(line)]
        if line_numbers:
            hits[phrase] = line_numbers
    return hits


def structural_stats(text, lines):
    words = text.split()
    word_count = max(len(words), 1)
    non_blank = [l for l in lines if l.strip()]
    bullet_lines = [l for l in non_blank if BULLET_RE.match(l)]
    heading_lines = [l for l in non_blank if HEADING_RE.match(l)]
    bold_hits = BOLD_RE.findall(text)
    prose_text = "\n".join(l for l in lines if not TABLE_SEPARATOR_RE.match(l))
    em_dashes = EM_DASH_RE.findall(prose_text)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip() and not HEADING_RE.match(p.strip())]

    return {
        "word_count": word_count,
        "bullet_ratio": len(bullet_lines) / max(len(non_blank), 1),
        "heading_count": len(heading_lines),
        "paragraph_count": max(len(paragraphs), 1),
        "bold_per_1000w": len(bold_hits) / word_count * 1000,
        "em_dash_per_1000w": len(em_dashes) / word_count * 1000,
    }


def check_file(path, banned_words):
    text = path.read_text()
    lines = text.splitlines()

    word_hits = find_word_hits(lines, banned_words)
    stats = structural_stats(text, lines)
    negative_parallelism = [
        i + 1 for i, l in enumerate(lines) if NEGATIVE_PARALLELISM_RE.search(l)
    ]

    print(f"\n=== {path} ===")
    fail = False

    if word_hits:
        fail = True
        print(f"\nBanned words ({len(word_hits)} distinct):")
        for phrase, line_numbers in sorted(word_hits.items()):
            print(f"  '{phrase}' — line(s) {', '.join(map(str, line_numbers))}")
    else:
        print("\nBanned words: none found.")

    if negative_parallelism:
        fail = True
        print(f"\nNegative parallelism ('not just X, but Y'): line(s) "
              f"{', '.join(map(str, negative_parallelism))}")

    print(f"\nStructural stats ({stats['word_count']} words, "
          f"{stats['paragraph_count']} paragraphs, {stats['heading_count']} headings):")
    warnings = []
    if stats["bullet_ratio"] > 0.35:
        warnings.append(f"  WARN bullet-line ratio {stats['bullet_ratio']:.0%} — "
                         f"check for bullet-ified prose that should be sentences")
    if stats["heading_count"] / stats["paragraph_count"] > 0.5:
        warnings.append(f"  WARN heading-to-paragraph ratio "
                         f"{stats['heading_count']}/{stats['paragraph_count']} — "
                         f"possible header-per-paragraph over-structuring")
    if stats["bold_per_1000w"] > 15:
        warnings.append(f"  WARN bold density {stats['bold_per_1000w']:.1f}/1000 words — "
                         f"check bold isn't applied to every jargon term on first use")
    if stats["em_dash_per_1000w"] > 3:
        warnings.append(f"  WARN em dash density {stats['em_dash_per_1000w']:.1f}/1000 words")

    if warnings:
        print("\n".join(warnings))
    else:
        print("  no structural warnings")

    return fail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="markdown file(s) to check")
    args = parser.parse_args()

    banned_words = load_banned_words(RULES_PATH)

    any_fail = False
    for path in args.paths:
        if check_file(path, banned_words):
            any_fail = True

    print()
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
