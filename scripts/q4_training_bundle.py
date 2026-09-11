"""Run up to four explicitly configured CPU experiments under one supervisor.

Inherits the existing pair launcher's signal, checkpoint grace and group cleanup
contract. Only the fixed Q4 trainer modules and their declared options are valid.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re

from scripts.q4_training_pair import (_identifier, _inside, INTEGER_FLAGS,
    FLOAT_FLAGS, ALLOWED_FLAGS, require_outer_supervisor, run_pair)


INITIALIZATION_FLAGS = {"--initialize-micro-warmstart", "--initialize-sha256"}
MODULE_FLAGS = {
    "q4_rl.train": ALLOWED_FLAGS,
    "q4_rl.micro_train": ALLOWED_FLAGS | INITIALIZATION_FLAGS | {"--architecture"},
    "q4_rl.scst_train": {"--output", "--workers", "--cpu-budget", "--batch-pairs",
        "--minibatch-size", "--learning-rate", "--max-decisions", "--entropy-coefficient",
        "--random-seed", "--scenario-start", "--scenario-end", "--max-batches",
        "--max-wall-seconds", "--deadline"} | INITIALIZATION_FLAGS,
}


def validate_config(config, root):
    if not isinstance(config, dict) or set(config) != {"run", "jobs"}:
        raise ValueError("configuration requires exactly run and jobs")
    run = _inside(root, "runs/"+_identifier(config["run"]))
    if not isinstance(config["jobs"], list) or not 1 <= len(config["jobs"]) <= 4:
        raise ValueError("one to four declared trainer jobs required")
    jobs, names, total_budget = [], set(), 0
    for job in config["jobs"]:
        if not isinstance(job, dict) or set(job) != {"name", "module", "argv"}:
            raise ValueError("job requires name, module and argv")
        name, module = _identifier(job["name"]), job["module"]
        if name in names or not isinstance(module, str) or module not in MODULE_FLAGS:
            raise ValueError("duplicate name or unsupported trainer module")
        argv = job["argv"]
        if (not isinstance(argv, list) or not argv or len(argv) % 2 or
                any(not isinstance(x, str) or not x or any(ord(c) < 32 for c in x) for x in argv)):
            raise ValueError("argv requires explicit option/value pairs")
        flags = {}
        for flag, value in zip(argv[::2], argv[1::2]):
            if flag not in MODULE_FLAGS[module] or flag in flags:
                raise ValueError("unknown, duplicate or forbidden trainer option")
            if flag in INTEGER_FLAGS | {"--batch-pairs"}:
                if not re.fullmatch(r"[0-9]+", value) or (
                        int(value) == 0 and flag not in {"--warmstart-episodes", "--random-seed"}):
                    raise ValueError("invalid integer option")
            elif flag in FLOAT_FLAGS:
                number = float(value)
                if not math.isfinite(number) or number < 0 or (number == 0 and flag != "--entropy-coefficient"):
                    raise ValueError("invalid numerical option")
            elif flag == "--architecture" and value not in {"mlp", "induced"}:
                raise ValueError("unknown micro architecture")
            elif flag == "--deadline" and datetime.fromisoformat(value).tzinfo is None:
                raise ValueError("deadline needs an explicit timezone")
            elif flag == "--initialize-sha256" and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("initialization needs a lowercase SHA256")
            elif flag == "--initialize-micro-warmstart":
                source = _inside(root, value)
                if source.suffix != ".pt" or not source.is_file():
                    raise ValueError("initialization must be an existing task checkpoint")
                value = source.relative_to(root).as_posix()
            flags[flag] = value
        if not {"--output", "--workers", "--cpu-budget"} <= flags.keys():
            raise ValueError("output and explicit worker/CPU budgets are required")
        output = _inside(root, flags["--output"])
        if output != run/name/"training" or (output.exists() and
                (not output.is_dir() or any(output.iterdir()))):
            raise ValueError("each fresh output must be runs/<run>/<job>/training")
        if bool(flags.get("--initialize-micro-warmstart")) != bool(flags.get("--initialize-sha256")):
            raise ValueError("initialization path and SHA must be specified together")
        budget = int(flags["--cpu-budget"])
        if not 1 <= int(flags["--workers"]) < budget <= 50:
            raise ValueError("workers plus learner must fit the CPU allowance")
        total_budget += budget
        flags["--output"] = output.relative_to(root).as_posix()
        jobs.append(dict(name=name, module=module, argv=[x for pair in flags.items() for x in pair],
                         output=flags["--output"]))
        names.add(name)
    if total_budget > 50:
        raise ValueError("combined sampling and learner allowances exceed 50 CPUs")
    return run, jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    path = _inside(root, args.config)
    run, jobs = validate_config(json.loads(path.read_text(encoding="utf-8-sig")), root)
    require_outer_supervisor(root)
    return run_pair(run, jobs, root)


if __name__ == "__main__":
    raise SystemExit(main())
