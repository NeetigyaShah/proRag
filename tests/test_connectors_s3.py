"""moto-backed tests for S3Connector (#22): listing/prefix filtering,
since-filtering, and fetch — no real AWS/MinIO/R2 credentials needed."""

import boto3
from moto import mock_aws

from prorag.connectors.s3 import S3Connector


def _client():
    return boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )


def _connector(bucket: str, prefix: str = "") -> S3Connector:
    return S3Connector(
        {
            "bucket": bucket,
            "prefix": prefix,
            "region": "us-east-1",
            "access_key_id": "testing",
            "secret_access_key": "testing",
        }
    )


@mock_aws
def test_list_changed_only_returns_objects_under_prefix():
    client = _client()
    client.create_bucket(Bucket="docs")
    client.put_object(Bucket="docs", Key="reports/a.pdf", Body=b"hello")
    client.put_object(Bucket="docs", Key="reports/b.txt", Body=b"world")
    client.put_object(Bucket="docs", Key="other/c.pdf", Body=b"nope")

    objs = _connector("docs", prefix="reports/").list_changed()

    assert {o.key for o in objs} == {"reports/a.pdf", "reports/b.txt"}
    for o in objs:
        assert o.etag
        assert o.size > 0
        assert o.last_modified is not None


@mock_aws
def test_list_all_ids_returns_every_key():
    client = _client()
    client.create_bucket(Bucket="docs")
    client.put_object(Bucket="docs", Key="a.pdf", Body=b"hello")
    client.put_object(Bucket="docs", Key="b.pdf", Body=b"world")

    assert set(_connector("docs").list_all_ids()) == {"a.pdf", "b.pdf"}


@mock_aws
def test_fetch_returns_object_bytes():
    client = _client()
    client.create_bucket(Bucket="docs")
    client.put_object(Bucket="docs", Key="a.txt", Body=b"the exact content")

    connector = _connector("docs")
    [obj] = connector.list_changed()

    assert connector.fetch(obj) == b"the exact content"


@mock_aws
def test_list_changed_since_filters_out_objects_not_touched_after_the_cutoff():
    from datetime import UTC, datetime, timedelta

    client = _client()
    client.create_bucket(Bucket="docs")
    client.put_object(Bucket="docs", Key="a.txt", Body=b"x")

    connector = _connector("docs")
    future_cutoff = datetime.now(UTC) + timedelta(days=1)

    assert connector.list_changed(since=future_cutoff) == []
    assert len(connector.list_changed(since=None)) == 1


@mock_aws
def test_list_changed_empty_bucket_returns_empty_list():
    client = _client()
    client.create_bucket(Bucket="docs")

    assert _connector("docs").list_changed() == []
