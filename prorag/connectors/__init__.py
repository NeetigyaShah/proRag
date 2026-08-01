"""Sync-engine connectors (#22, per #15's polling-first / per-connector
fidelity-tier architecture and #6's ranking — S3/Blob is the plumbing
connector, cheapest to stand up, and Tier C: no source ACLs exist to mirror).

Layout: s3.py (the S3Connector implementation), sync.py (sync_incremental /
full_sweep — plain functions over any connector, testable without HTTP),
router.py (admin-only CRUD + POST /connectors/{id}/sync).
"""
