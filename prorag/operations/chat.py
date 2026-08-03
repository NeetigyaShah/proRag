"""Operations layer — chat persistence.

persist_exchange() is the single place a user turn + assistant turn (plus one
Citation row per actually-cited source) are written, shared by POST /chat and
POST /chat/stream. Splitting it out of chat/router.py lets both endpoints
share one persistence path and keeps the router HTTP-only.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from prorag.models import Chat, Citation, Message
from prorag.schemas import Source


async def persist_exchange(
    session: AsyncSession,
    chat_id: uuid.UUID | None,
    user_message: str,
    answer_text: str,
    cited_sources: list[Source],
) -> tuple[uuid.UUID, uuid.UUID]:
    """Persists the user turn + assistant turn + one citations row per
    actually-cited source (§5.3 — the writer QDMS-AI's `cited_sources`
    column never had). Returns (chat_id, assistant_message_id)."""
    if chat_id is None:
        chat = Chat()
        session.add(chat)
        await session.flush()
        chat_id = chat.id

    session.add(Message(chat_id=chat_id, role="user", content=user_message))

    assistant_message = Message(chat_id=chat_id, role="assistant", content=answer_text)
    session.add(assistant_message)
    await session.flush()

    for s in cited_sources:
        session.add(Citation(message_id=assistant_message.id, n=s.n, doc_id=s.doc_id, page=s.page, bbox=s.bbox))

    await session.commit()
    return chat_id, assistant_message.id
