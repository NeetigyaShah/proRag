"""Pure diff-logic tests for the sync engine (#22): new/changed/unchanged/
deleted matrix, no DB and no S3 fixture — diff_objects()/find_deleted() are
plain functions over in-memory data precisely so this doesn't need either."""

from datetime import UTC, datetime

from prorag.connectors.s3 import RemoteObject
from prorag.connectors.sync import diff_objects, find_deleted


def _obj(key: str, etag: str = "e1", size: int = 10) -> RemoteObject:
    return RemoteObject(key=key, etag=etag, size=size, last_modified=datetime.now(UTC))


def test_diff_objects_classifies_new_changed_unchanged():
    remote = [_obj("a.txt", "e1", 10), _obj("b.txt", "e2", 20), _obj("c.txt", "e3", 30)]
    known = {"b.txt": ("e2", 20), "c.txt": ("old-etag", 30)}  # a: unseen: b: identical; c: etag changed

    new, changed, unchanged = diff_objects(remote, known)

    assert [o.key for o in new] == ["a.txt"]
    assert [o.key for o in changed] == ["c.txt"]
    assert unchanged == ["b.txt"]


def test_diff_objects_size_change_alone_counts_as_changed():
    remote = [_obj("a.txt", "e1", 99)]
    known = {"a.txt": ("e1", 10)}  # same etag, different size — still a change

    new, changed, unchanged = diff_objects(remote, known)

    assert new == []
    assert [o.key for o in changed] == ["a.txt"]
    assert unchanged == []


def test_diff_objects_identical_etag_and_size_is_unchanged():
    remote = [_obj("a.txt", "e1", 10)]
    known = {"a.txt": ("e1", 10)}

    new, changed, unchanged = diff_objects(remote, known)

    assert new == changed == []
    assert unchanged == ["a.txt"]


def test_diff_objects_empty_inputs():
    assert diff_objects([], {}) == ([], [], [])


def test_find_deleted_returns_keys_missing_from_the_listing():
    known = {"a.txt": "synced", "b.txt": "synced", "c.txt": "skipped"}
    assert find_deleted({"a.txt"}, known) == {"b.txt", "c.txt"}


def test_find_deleted_ignores_items_already_marked_deleted():
    """A previous sweep already deleted this one — it shouldn't be
    rediscovered as newly-missing on every subsequent sweep."""
    known = {"a.txt": "deleted"}
    assert find_deleted(set(), known) == set()


def test_find_deleted_empty_when_everything_still_present():
    known = {"a.txt": "synced", "b.txt": "skipped"}
    assert find_deleted({"a.txt", "b.txt"}, known) == set()


def test_find_deleted_empty_known_set():
    assert find_deleted({"a.txt"}, {}) == set()
