"""End-to-end sync test (#22): a moto-backed S3 bucket synced through
sync_incremental/full_sweep, all the way through ingest_bytes into real
Postgres, with embeddings monkeypatched to random vectors (no live LLM
provider needed — same idea as tests/test_retrieval.py's embed_texts stub).

DB-test pattern matches tests/test_identity_schema.py: skip cleanly if
Postgres is unreachable, clean up rows in a `finally`. tests/conftest.py's
autouse fixture disposes the engine after every test (#23), so this no
longer does it itself.
"""

import random
import uuid

import boto3
import pytest
from moto import mock_aws
from sqlalchemy import delete, select, text

import prorag.ingest.core as ingest_core
from prorag.connectors.sync import full_sweep, sync_incremental
from prorag.db import SessionLocal
from prorag.models import Connector, ConnectorItem, Document
from prorag.settings import settings


async def _get_session():
    try:
        session = SessionLocal()
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    return session


async def _fake_embed(texts, session=None):
    return [[random.random() for _ in range(settings.embed_dim)] for _ in texts]


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch):
    monkeypatch.setattr(ingest_core, "embed_texts_batched", _fake_embed)


def _client():
    return boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )


def _s3_config(bucket: str) -> dict:
    return {
        "bucket": bucket,
        "region": "us-east-1",
        "access_key_id": "testing",
        "secret_access_key": "testing",
    }


async def _cleanup(session, connector: Connector) -> None:
    items = (
        await session.execute(select(ConnectorItem).where(ConnectorItem.connector_id == connector.id))
    ).scalars().all()
    doc_ids = [i.doc_id for i in items if i.doc_id is not None]
    await session.execute(delete(ConnectorItem).where(ConnectorItem.connector_id == connector.id))
    await session.execute(delete(Connector).where(Connector.id == connector.id))
    if doc_ids:
        await session.execute(delete(Document).where(Document.id.in_(doc_ids)))
    await session.commit()


async def test_sync_incremental_ingests_new_skips_unsupported_and_is_idempotent():
    with mock_aws():
        client = _client()
        bucket = f"bucket-{uuid.uuid4().hex[:8]}"
        client.create_bucket(Bucket=bucket)
        client.put_object(Bucket=bucket, Key="a.txt", Body=b"hello world, this is a test document about widgets.")
        client.put_object(Bucket=bucket, Key="ignore.exe", Body=b"binary junk, not a supported document type at all.")

        session = await _get_session()
        tag = uuid.uuid4().hex[:8]
        async with session:
            connector = Connector(id=uuid.uuid4(), type="s3", name=f"s3-{tag}", config=_s3_config(bucket))
            session.add(connector)
            await session.commit()

            try:
                report = await sync_incremental(connector, session)
                assert report == {"new": 1, "changed": 0, "skipped": 1, "errors": 0, "deleted": 0}

                items = (
                    await session.execute(select(ConnectorItem).where(ConnectorItem.connector_id == connector.id))
                ).scalars().all()
                assert {i.external_id for i in items} == {"a.txt", "ignore.exe"}

                synced = next(i for i in items if i.external_id == "a.txt")
                assert synced.status == "synced"
                assert synced.doc_id is not None

                skipped = next(i for i in items if i.external_id == "ignore.exe")
                assert skipped.status == "skipped"
                assert skipped.doc_id is None
                assert skipped.error == "unsupported file type"

                doc = (await session.execute(select(Document).where(Document.id == synced.doc_id))).scalar_one()
                assert doc.status == "ready"

                # Re-syncing with nothing changed on the remote side re-ingests nothing.
                report2 = await sync_incremental(connector, session)
                assert report2 == {"new": 0, "changed": 0, "skipped": 0, "errors": 0, "deleted": 0}

                # Changing the object's content is picked up as 'changed'.
                client.put_object(
                    Bucket=bucket, Key="a.txt", Body=b"hello world, this is an updated widget document now."
                )
                report3 = await sync_incremental(connector, session)
                assert report3["changed"] == 1
                assert report3["new"] == 0
            finally:
                await _cleanup(session, connector)


async def test_full_sweep_propagates_deletion_and_cascades_the_document():
    with mock_aws():
        client = _client()
        bucket = f"bucket-{uuid.uuid4().hex[:8]}"
        client.create_bucket(Bucket=bucket)
        client.put_object(Bucket=bucket, Key="keep.txt", Body=b"a document about widgets and gadgets and other things.")
        client.put_object(Bucket=bucket, Key="remove.txt", Body=b"a document about sprockets and other stuff entirely.")

        session = await _get_session()
        tag = uuid.uuid4().hex[:8]
        async with session:
            connector = Connector(id=uuid.uuid4(), type="s3", name=f"s3-{tag}", config=_s3_config(bucket))
            session.add(connector)
            await session.commit()

            try:
                report = await full_sweep(connector, session)
                assert report["new"] == 2
                assert report["deleted"] == 0

                items = (
                    await session.execute(select(ConnectorItem).where(ConnectorItem.connector_id == connector.id))
                ).scalars().all()
                removed_doc_id = next(i for i in items if i.external_id == "remove.txt").doc_id
                assert removed_doc_id is not None

                client.delete_object(Bucket=bucket, Key="remove.txt")

                report2 = await full_sweep(connector, session)
                assert report2["deleted"] == 1
                assert report2["new"] == 0

                removed_doc = (
                    await session.execute(select(Document).where(Document.id == removed_doc_id))
                ).scalar_one_or_none()
                assert removed_doc is None  # cascade-deleted, per Document's relationships

                removed_item = (
                    await session.execute(
                        select(ConnectorItem).where(
                            ConnectorItem.connector_id == connector.id, ConnectorItem.external_id == "remove.txt"
                        )
                    )
                ).scalar_one()
                assert removed_item.status == "deleted"
                assert removed_item.doc_id is None

                kept_item = (
                    await session.execute(
                        select(ConnectorItem).where(
                            ConnectorItem.connector_id == connector.id, ConnectorItem.external_id == "keep.txt"
                        )
                    )
                ).scalar_one()
                assert kept_item.status == "synced"

                # Sweeping again doesn't rediscover the already-deleted item.
                report3 = await full_sweep(connector, session)
                assert report3["deleted"] == 0
            finally:
                await _cleanup(session, connector)


async def test_sync_dedup_links_connector_item_to_existing_document_without_reingesting():
    """Two keys with byte-identical content share a sha256 — ingest_bytes'
    existing dedup path resolves the second to the first's doc_id instead of
    re-ingesting (#22's dedup-interplay requirement)."""
    with mock_aws():
        client = _client()
        bucket = f"bucket-{uuid.uuid4().hex[:8]}"
        client.create_bucket(Bucket=bucket)
        body = b"identical content shared by two different keys, for the dedup path test."
        client.put_object(Bucket=bucket, Key="one.txt", Body=body)
        client.put_object(Bucket=bucket, Key="two.txt", Body=body)

        session = await _get_session()
        tag = uuid.uuid4().hex[:8]
        async with session:
            connector = Connector(id=uuid.uuid4(), type="s3", name=f"s3-{tag}", config=_s3_config(bucket))
            session.add(connector)
            await session.commit()

            try:
                report = await sync_incremental(connector, session)
                assert report["errors"] == 0

                items = (
                    await session.execute(select(ConnectorItem).where(ConnectorItem.connector_id == connector.id))
                ).scalars().all()
                doc_ids = {i.doc_id for i in items}
                assert len(doc_ids) == 1  # both keys resolve to the same document
                assert all(i.status == "synced" for i in items)
            finally:
                await _cleanup(session, connector)


async def test_sync_records_per_item_error_and_continues_the_run():
    """A too-large object errors out (over max_upload_bytes) but doesn't
    stop the rest of the batch from syncing."""
    with mock_aws():
        client = _client()
        bucket = f"bucket-{uuid.uuid4().hex[:8]}"
        client.create_bucket(Bucket=bucket)
        client.put_object(Bucket=bucket, Key="good.txt", Body=b"a perfectly normal small document about widgets.")
        client.put_object(Bucket=bucket, Key="huge.txt", Body=b"x" * 1024)

        session = await _get_session()
        tag = uuid.uuid4().hex[:8]
        async with session:
            connector = Connector(id=uuid.uuid4(), type="s3", name=f"s3-{tag}", config=_s3_config(bucket))
            session.add(connector)
            await session.commit()

            try:
                original_max = settings.max_upload_bytes
                try:
                    settings.max_upload_bytes = 100  # smaller than huge.txt, larger than good.txt
                    report = await sync_incremental(connector, session)
                finally:
                    settings.max_upload_bytes = original_max

                assert report["new"] == 1
                assert report["skipped"] == 1

                items = (
                    await session.execute(select(ConnectorItem).where(ConnectorItem.connector_id == connector.id))
                ).scalars().all()
                huge_item = next(i for i in items if i.external_id == "huge.txt")
                assert huge_item.status == "skipped"
                assert huge_item.error == "exceeds max_upload_bytes"
            finally:
                await _cleanup(session, connector)
