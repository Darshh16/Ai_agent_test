"""
Parses knowledge-base/*.md into heading-level chunks with metadata preserved
from YAML front matter.

Design notes:
- Chunking is by heading (##), not fixed character windows, so every chunk
  maps to a real, citable section of a real document.
- We do NOT filter out non-authoritative documents (e.g. the draft migration
  notes file) at ingest time. Everything gets indexed with its metadata
  intact. Authority filtering happens at retrieval/prompting time, not here
  -- because a user can still reference a non-authoritative document by name
  ("the migration note says...") and the agent needs to be able to see it in
  order to correctly reject it, rather than being unable to respond to the
  claim at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    filename: str
    document_id: str
    title: str
    heading: str
    text: str
    status: str
    policy_authority: str
    audience: str
    supersedes: str | None = None
    superseded_by: str | None = None
    metadata: dict = field(default_factory=dict)

    def is_citable_authority(self) -> bool:
        """A chunk may be cited as customer-facing policy authority only if
        it is both active and officially authoritative. Superseded, draft,
        or non-official content must never be presented as current policy,
        even if it gets retrieved and shown to the model for context."""
        return self.status == "active" and self.policy_authority == "official"

    def source_label(self) -> str:
        return f"{self.filename} — {self.heading}" if self.heading else self.filename


def _parse_front_matter(raw_text: str) -> tuple[dict, str]:
    match = FRONT_MATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text
    front_matter_raw, body = match.groups()
    metadata = yaml.safe_load(front_matter_raw) or {}
    return metadata, body


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (heading, section_text) pairs on '## ' headings.
    Any content before the first '##' heading (e.g. the H1 title / intro line)
    is kept as its own section with an empty heading."""
    positions = [m.start() for m in HEADING_RE.finditer(body)]
    sections: list[tuple[str, str]] = []

    if not positions:
        return [("", body.strip())]

    if positions[0] > 0:
        intro = body[: positions[0]].strip()
        if intro:
            sections.append(("", intro))

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(body)
        section_text = body[start:end].strip()
        heading_match = HEADING_RE.match(section_text)
        heading = heading_match.group(1).strip() if heading_match else ""
        sections.append((heading, section_text))

    return sections


def load_knowledge_base(kb_dir: str | Path) -> list[Chunk]:
    kb_dir = Path(kb_dir)
    chunks: list[Chunk] = []

    for filepath in sorted(kb_dir.glob("*.md")):
        raw_text = filepath.read_text(encoding="utf-8")
        metadata, body = _parse_front_matter(raw_text)

        document_id = metadata.get("document_id", filepath.stem)
        title = metadata.get("title", filepath.stem)
        status = metadata.get("status", "unknown")
        policy_authority = metadata.get("policy_authority", "unknown")
        audience = metadata.get("audience", "unknown")

        sections = _split_into_sections(body)
        for idx, (heading, section_text) in enumerate(sections):
            # Skip empty/whitespace-only sections (e.g. a heading with no
            # body written under it yet in the source doc).
            if not section_text or len(section_text.strip()) < 3:
                continue

            chunk = Chunk(
                chunk_id=f"{filepath.stem}::{idx}",
                filename=filepath.name,
                document_id=document_id,
                title=title,
                heading=heading or title,
                text=section_text,
                status=status,
                policy_authority=policy_authority,
                audience=audience,
                supersedes=metadata.get("supersedes"),
                superseded_by=metadata.get("superseded_by"),
                metadata=metadata,
            )
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    import sys

    kb_path = sys.argv[1] if len(sys.argv) > 1 else "knowledge-base"
    all_chunks = load_knowledge_base(kb_path)
    print(f"Loaded {len(all_chunks)} chunks from {kb_path}\n")
    for c in all_chunks:
        authority = "AUTHORITY" if c.is_citable_authority() else "no-authority"
        print(f"[{authority:12}] {c.source_label():65} status={c.status:11} policy_authority={c.policy_authority}")