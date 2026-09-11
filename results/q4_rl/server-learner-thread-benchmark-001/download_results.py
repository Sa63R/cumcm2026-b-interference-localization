"""Read this independent benchmark's object results; no SSH file transport."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK = "q4-rl-cpu-microbench-20260912"
spec = importlib.util.spec_from_file_location("exchange", ROOT/"scripts/q4_object_exchange.py")
exchange = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exchange)


def main():
    client = exchange.connection(ROOT.parent/"AGENTS.md")
    bucket, prefix = "bucket-c20250204-pool01", "lianghao/bwc/shumo/"+TASK+"/"
    objects = exchange.list_all_objects(client, bucket, prefix, "results/")
    if not objects:
        raise ValueError("Results not yet synchronized")
    output = HERE/"readback"
    output.mkdir(exist_ok=False)
    manifest = []
    for item in objects:
        name = item["name"].removeprefix("results/")
        target = output/name
        target.resolve().relative_to(output.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        response = client.get_object(Bucket=bucket, Key=prefix+item["name"])
        checksum = hashlib.sha256()
        with target.open("xb") as stream:
            try:
                for block in iter(lambda: response["Body"].read(1024*1024), b""):
                    checksum.update(block)
                    stream.write(block)
            finally:
                response["Body"].close()
        assert target.stat().st_size == item["bytes"] == response["ContentLength"]
        manifest.append(dict(name=name, bytes=item["bytes"], sha256=checksum.hexdigest(), etag=response.get("ETag")))
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(output/"incoming/source.tar.gz") == "c37ddfed148a4713812d48c70d9501e9e6104e23fd8c93ee4d99712b611b39d9"
    assert sha(output/"inputs/warmstart.pt") == "61d4ac338cdfc0b62fcbb0cf9c4f2baaf9f832c8d203ac8d33a0f9b308f5ce49"
    timing = json.loads((output/"learner_thread_benchmark/timings.json").read_text())
    assert sha(output/"learner_thread_benchmark/run.py") == timing["script_sha256"] == "af85b9ff66ee99d8b2d136d481a1657fedcb566cc4f113ccbe4022e8e124787f"
    with tarfile.open(output/"incoming/source.tar.gz") as archive:
        for name, expected in timing["source_sha256"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
    for result in timing["results"]:
        assert sha(output/"inputs/batch-000000-attempt-000000.json.gz") == result["source"]["index_sha256"]
        for item in result["source"]["episodes"]:
            assert sha(output/"inputs"/item["file"]) == item["sha256"]
    (HERE/"OBJECT_READBACK.json").write_text(json.dumps(dict(task=TASK, objects=manifest,
        source_input_and_script_hashes_verified=True), indent=2), encoding="utf-8")
    print(json.dumps(dict(task=TASK, files=len(manifest), bytes=sum(item["bytes"] for item in manifest),
        comparisons=timing["comparisons"], total_worker_cpu_s=timing["total_worker_cpu_s"])))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(error_type=type(error).__name__)))
        raise SystemExit(1)
