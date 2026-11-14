from __future__ import annotations

import hashlib
from uuid import uuid4

from helpdesk.domain.documents import ChunkId, DocumentId, VersionId
from helpdesk.domain.retrieval import Citation

FINGERPRINT = hashlib.sha256(b"question").hexdigest()


def chunk_id() -> ChunkId:
    return ChunkId(uuid4())


def citation(cited: ChunkId | None = None, *, title: str = "115 碩專簡章") -> Citation:
    return Citation(
        chunk_id=cited or chunk_id(),
        document_id=DocumentId(uuid4()),
        version_id=VersionId(uuid4()),
        title=title,
        section_path="報名資格",
        page_number=3,
    )
