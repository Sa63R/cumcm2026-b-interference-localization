"""Exchange files inside one explicit S3 prefix; credentials remain external."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Private JSON outside the code/package")
    parser.add_argument("--prefix", help="Override the dedicated task prefix without editing credentials")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe")
    sub.add_parser("list")
    for command in ("upload", "download"):
        child = sub.add_parser(command)
        child.add_argument("file", type=Path)
        child.add_argument("--name", help="Relative key within the dedicated exchange prefix")
    args = parser.parse_args()
    conf = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.prefix is not None:
        conf["prefix"] = args.prefix
    if not conf["endpoint"].startswith("https://") or not conf["prefix"].strip("/"):
        raise ValueError("HTTPS and a nonempty task prefix are required")
    bucket, prefix = conf["bucket"], conf["prefix"].strip("/")+"/"
    client = boto3.client("s3", endpoint_url=conf["endpoint"],
        aws_access_key_id=conf["access_key_id"], aws_secret_access_key=conf["secret_access_key"],
        region_name=conf.get("region", "us-east-1"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      request_checksum_calculation="when_required", response_checksum_validation="when_required",
                      connect_timeout=10, read_timeout=60, retries={"max_attempts": 3}))
    if args.command in ("probe", "list"):
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1 if args.command=="probe" else 100)
        print(json.dumps({"ok": True, "prefix": prefix,
                          "files": [{"name": x["Key"][len(prefix):], "bytes": x["Size"]}
                                    for x in response.get("Contents", [])] if args.command=="list" else None}))
        return
    name = args.name or args.file.name
    key_path = PurePosixPath(name)
    if not name or not key_path.parts or key_path.is_absolute() or any(p in {"..", "."} for p in key_path.parts) or "\\" in name:
        raise ValueError("Expected a relative key inside the task prefix")
    key = prefix+name
    if args.command == "upload":
        sha = hashlib.sha256(args.file.read_bytes()).hexdigest()
        try:
            existing = client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            if existing.get("Metadata", {}).get("sha256") == sha:
                print(json.dumps({"ok": True, "already_uploaded": True, "key": key, "sha256": sha}))
                return
            raise ValueError("Object already exists with different contents; choose a new name")
        client.upload_file(str(args.file), bucket, key, ExtraArgs={"Metadata": {"sha256": sha}})
        remote = client.head_object(Bucket=bucket, Key=key)
        if remote["ContentLength"] != args.file.stat().st_size or remote.get("Metadata", {}).get("sha256") != sha:
            raise ValueError("Uploaded metadata/size verification failed")
        # Download checksum independently verifies actual bytes, not merely metadata.
        stream = client.get_object(Bucket=bucket, Key=key)["Body"]
        digest = hashlib.sha256()
        try:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        finally:
            stream.close()
        if digest.hexdigest() != sha:
            raise ValueError("Uploaded bytes checksum verification failed")
        print(json.dumps({"ok": True, "key": key, "sha256": sha, "bytes": args.file.stat().st_size}))
    else:
        if args.file.exists():
            raise ValueError("Local target exists; download into a new filename")
        args.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.file.with_suffix(args.file.suffix+".part")
        client.download_file(bucket, key, str(tmp))
        remote = client.head_object(Bucket=bucket, Key=key)
        sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if remote.get("Metadata", {}).get("sha256") not in (None, sha):
            raise ValueError("Downloaded file checksum mismatch")
        tmp.replace(args.file)
        print(json.dumps({"ok": True, "sha256": sha, "bytes": args.file.stat().st_size}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "service_code": code}))
        raise SystemExit(1)
