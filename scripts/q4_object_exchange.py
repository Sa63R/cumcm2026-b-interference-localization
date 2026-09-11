"""Local object exchange; credentials are read in memory from an external file.

Never print credential values or include the credentials document in a release.
The remote side uses its existing rclone configuration instead.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

BASE = "lianghao/bwc/shumo/"


def list_all_objects(client, bucket, task_prefix, name_prefix=""):
    """Consume every continuation page or fail explicitly on a broken cursor."""
    result, cursor, seen_cursors, seen_keys = [], None, set(), set()
    prefix = task_prefix + name_prefix
    while True:
        request = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 200}
        if cursor is not None:
            request["ContinuationToken"] = cursor
        page = client.list_objects_v2(**request)
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.startswith(prefix):
                raise ValueError("Object listing escaped requested prefix")
            if key not in seen_keys:
                result.append({"name": key[len(task_prefix):], "bytes": item["Size"]})
                seen_keys.add(key)
        if not page.get("IsTruncated", False):
            return result
        cursor = page.get("NextContinuationToken")
        if not cursor or cursor in seen_cursors:
            raise ValueError("Truncated object listing has no advancing continuation token")
        seen_cursors.add(cursor)


def connection(document):
    import boto3
    from botocore.config import Config
    content = Path(document).read_text(encoding="utf-8-sig").replace("\\_", "_")
    values = {}
    for field in ("access_key_id", "secret_access_key", "endpoint"):
        match = re.search(r"(?m)^\s*" + field + r"\s*=\s*(\S+)\s*$", content)
        if match is None:
            raise ValueError("Missing external object-store configuration")
        values[field] = match.group(1)
    if not values["endpoint"].startswith("https://"):
        raise ValueError("HTTPS is required")
    return boto3.client("s3", endpoint_url=values["endpoint"],
        aws_access_key_id=values["access_key_id"], aws_secret_access_key=values["secret_access_key"],
        region_name="us-east-1", config=Config(signature_version="s3v4",
        s3={"addressing_style": "path"}, connect_timeout=10, read_timeout=60,
        retries={"max_attempts": 3}, request_checksum_calculation="when_required",
        response_checksum_validation="when_required"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-document", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("action", choices=("upload", "download", "list"))
    parser.add_argument("--file", type=Path)
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    if not re.fullmatch(r"q4-[A-Za-z0-9_-]+", args.task):
        raise ValueError("An independent Q4 task prefix is required")
    name = PurePosixPath(args.name)
    if name.is_absolute() or ".." in name.parts or "\\" in args.name:
        raise ValueError("Object name must remain under the task prefix")
    client = connection(args.credentials_document)
    bucket, prefix = "bucket-c20250204-pool01", BASE + args.task + "/"
    if args.action == "list":
        print(json.dumps(list_all_objects(client, bucket, prefix, args.name)))
        return
    if not args.name or not args.file:
        raise ValueError("file and name required")
    key = prefix + args.name
    if args.action == "upload":
        from botocore.exceptions import ClientError
        sha = hashlib.file_digest(args.file.open("rb"), "sha256").hexdigest()
        try:
            old = client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            if old.get("Metadata", {}).get("sha256") == sha:
                print(json.dumps({"already_uploaded": True, "name": args.name, "sha256": sha}))
                return
            raise ValueError("Immutable release name already exists")
        client.upload_file(str(args.file), bucket, key, ExtraArgs={"Metadata": {"sha256": sha}})
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
        checksum = hashlib.sha256()
        try:
            for block in iter(lambda: body.read(1024 * 1024), b""):
                checksum.update(block)
        finally:
            body.close()
        if checksum.hexdigest() != sha:
            raise ValueError("Round-trip checksum mismatch")
        print(json.dumps({"uploaded": True, "name": args.name, "sha256": sha,
                          "bytes": args.file.stat().st_size}))
    else:
        if args.file.exists():
            raise ValueError("Download target exists")
        args.file.parent.mkdir(parents=True, exist_ok=True)
        temp = args.file.with_suffix(args.file.suffix + ".part")
        client.download_file(bucket, key, str(temp))
        with temp.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = client.head_object(Bucket=bucket, Key=key).get("Metadata", {}).get("sha256")
        if expected and expected != sha:
            raise ValueError("Download checksum mismatch")
        temp.replace(args.file)
        print(json.dumps({"downloaded": True, "name": args.name, "sha256": sha}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Service exceptions may carry headers or URLs. Do not echo their text.
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))
        raise SystemExit(1)
