"""組裝證據：從 RRF 排序的候選選出證據，依能否外送分成兩堆。

規則依序是：依分數穩定排序；同樣的內容在同一個外送類別只留分數最高的一段（可外送的副本不會
被分數更高的受限副本擋掉）；每份文件最多 `max_chunks_per_document` 段；token 預算只計算可外送
的證據，放不下的那段跳過、繼續看後面較短的；選滿 `max_evidence` 段為止。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from helpdesk.application.retrieval.ports import StoredChunk
from helpdesk.domain.documents import ChunkId, Document, DocumentId
from helpdesk.domain.retrieval import Candidate, Citation, Evidence, EvidenceSet


@dataclass(frozen=True, slots=True)
class AssemblyLimits:
    max_evidence: int
    max_chunks_per_document: int
    max_context_tokens: int


def assemble(
    candidates: Sequence[Candidate],
    chunks: Mapping[ChunkId, StoredChunk],
    sendable: Callable[[Document], bool],
    limits: AssemblyLimits,
) -> EvidenceSet:
    ranked = sorted(candidates, key=lambda candidate: candidate.fused_score, reverse=True)
    seen: set[tuple[str, bool]] = set()
    per_document: Counter[DocumentId] = Counter()
    remaining_tokens = limits.max_context_tokens
    piles: dict[bool, list[Evidence]] = {True: [], False: []}
    selected = 0
    for candidate in ranked:
        if selected >= limits.max_evidence:
            break
        stored = chunks[candidate.chunk_id]
        chunk, document = stored.chunk, stored.document
        external = sendable(document)
        identity = (chunk.content, external)
        if identity in seen or per_document[document.id] >= limits.max_chunks_per_document:
            continue
        if external and chunk.token_count > remaining_tokens:
            continue
        selected += 1
        piles[external].append(
            Evidence(
                rank=selected,
                citation=Citation(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    version_id=chunk.version_id,
                    title=document.metadata.title,
                    section_path=chunk.section_path,
                    page_number=chunk.page_number,
                ),
                content=chunk.content,
                document_type=document.metadata.document_type,
                token_count=chunk.token_count,
                score=candidate.fused_score,
            )
        )
        seen.add(identity)
        per_document[document.id] += 1
        if external:
            remaining_tokens -= chunk.token_count
    return EvidenceSet(sendable=tuple(piles[True]), restricted=tuple(piles[False]))
