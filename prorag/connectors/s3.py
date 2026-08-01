"""S3-compatible object storage connector (#22) — the plumbing connector per
#6: cheapest admin setup, no source ACLs to mirror (Tier C, #15). Works
against any S3-compatible endpoint (AWS, MinIO, Cloudflare R2) via boto3's
endpoint_url override.

Item identity is the object key; the change signal is etag+size (S3's ETag
is the content hash for non-multipart uploads and still changes on any
content edit for multipart ones) — see sync.py for how that's diffed against
connector_items. last_modified is carried along for visibility/`since`
filtering only; S3-compatible providers' clocks aren't trustworthy enough to
lean on alone.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

import boto3


@dataclass(frozen=True)
class RemoteObject:
    key: str
    etag: str
    size: int
    last_modified: datetime


class S3Connector:
    """list_changed()/list_all_ids() both paginate the full bucket+prefix
    listing — S3 has no server-side "changed since" filter that behaves
    identically across AWS/MinIO/R2, so `since` is applied client-side over
    the full listing rather than requested from the API."""

    def __init__(self, config: dict):
        self.bucket = config["bucket"]
        self.prefix = config.get("prefix", "")
        self._client = boto3.client(
            "s3",
            endpoint_url=config.get("endpoint_url") or None,
            aws_access_key_id=config.get("access_key_id"),
            aws_secret_access_key=config.get("secret_access_key"),
            region_name=config.get("region") or None,
        )

    def _paginate(self) -> Iterator[dict]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            yield from page.get("Contents", [])

    def list_changed(self, since: datetime | None = None) -> list[RemoteObject]:
        objs = [
            RemoteObject(key=o["Key"], etag=o["ETag"].strip('"'), size=o["Size"], last_modified=o["LastModified"])
            for o in self._paginate()
        ]
        if since is not None:
            objs = [o for o in objs if o.last_modified > since]
        return objs

    def list_all_ids(self) -> list[str]:
        return [o["Key"] for o in self._paginate()]

    def fetch(self, item: RemoteObject) -> bytes:
        resp = self._client.get_object(Bucket=self.bucket, Key=item.key)
        return resp["Body"].read()
