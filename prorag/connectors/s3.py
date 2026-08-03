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
    """One object from a bucket listing: the key plus the etag+size identity
    pair that sync.py diffs against connector_items, and last_modified carried
    for visibility/`since` filtering only."""

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
        """When called: once per sync run — sync.py's _build_connector()
        constructs this from the connector row's stored config. What: reads
        bucket/prefix/credentials and builds the boto3 S3 client (endpoint_url
        makes AWS/MinIO/R2 all work). Returns: None."""
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
        """When called: by list_changed and list_all_ids to walk the full
        bucket+prefix listing. What: pages through list_objects_v2, yielding
        each object's raw Contents dict. Returns: an iterator of dicts."""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            yield from page.get("Contents", [])

    def list_changed(self, since: datetime | None = None) -> list[RemoteObject]:
        """When called: by sync_incremental for the incremental poll (and by
        full_sweep with since=None to list everything). What: lists the full
        bucket+prefix and applies the `since` filter client-side on
        last_modified. Returns: the matching RemoteObjects."""
        objs = [
            RemoteObject(key=o["Key"], etag=o["ETag"].strip('"'), size=o["Size"], last_modified=o["LastModified"])
            for o in self._paginate()
        ]
        if since is not None:
            objs = [o for o in objs if o.last_modified > since]
        return objs

    def list_all_ids(self) -> list[str]:
        """When called: by callers that need just the full key universe — the
        sync engine itself uses list_changed; this is exercised by tests.
        What: lists every object key in the bucket+prefix. Returns: the list
        of key strings."""
        return [o["Key"] for o in self._paginate()]

    def fetch(self, item: RemoteObject) -> bytes:
        """When called: by sync.py's _process_objects for each new/changed
        object, run on a worker thread via asyncio.to_thread so the download
        doesn't stall the event loop. What: downloads the object's full body.
        Returns: the object bytes."""
        resp = self._client.get_object(Bucket=self.bucket, Key=item.key)
        return resp["Body"].read()
