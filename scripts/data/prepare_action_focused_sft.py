"""CPU-only Action-focused SFT data preparation.

Never executes teacher commands or loads weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import shlex
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CATEGORIES = ("bug-fix", "feature-request", "enhancement", "other")
ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_value(value):
    return json.loads(value) if isinstance(value, str) else value


def task_key(repo, instance_id=None, pr_number=None):
    repo = repo.strip().lower()
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo):
        raise ValueError(f"Invalid repository: {repo!r}")
    if pr_number is None:
        match = re.search(r"(?:_pr|[-_])(\d+)$", instance_id or "", re.I)
        if not match:
            raise ValueError(f"Cannot extract PR: {instance_id!r}")
        pr_number = match[1]
    return f"{repo}#{int(pr_number)}"


def terminal_candidate(command):
    if not isinstance(command, str):
        return False
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    return any(
        token == "echo" and tokens[index + 1] == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        for index, token in enumerate(tokens[:-1])
    )


def normalize(row):
    """Decode each JSON layer exactly once; preserve historical invalid schema calls."""
    from jsonschema import validators

    messages = copy.deepcopy(json_value(row["messages"]))
    tools = [json_value(t) for t in json_value(row["tools"])]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    schemas = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Unsupported tool schema")
        fn = tool["function"]
        schema = fn["parameters"]
        validators.validator_for(schema).check_schema(schema)
        if fn["name"] in schemas:
            raise ValueError("Duplicate tool definition")
        schemas[fn["name"]] = schema
    checks, pending, seen = [], [], set()
    for mi, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message {mi} must be an object")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Unsupported role: {role}")
        if role == "system" and mi != 0:
            raise ValueError("System message must be first")
        if message.get("content") is not None and not isinstance(message["content"], str):
            raise ValueError("Text-only content required")
        if role != "tool" and pending:
            raise ValueError(f"Missing observations before message {mi}")
        if role == "tool":
            if not pending:
                raise ValueError(f"orphan observation at {mi}")
            call_id = message.get("tool_call_id")
            if call_id:
                if call_id not in pending:
                    raise ValueError(f"Unknown observation id: {call_id}")
                pending.remove(call_id)
            else:
                pending.pop(0)
                checks.append(
                    {
                        "kind": "observation_link",
                        "status": "unverified",
                        "message_index": mi,
                        "reason": "No tool_call_id; positional only",
                    }
                )
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        if calls and role != "assistant":
            raise ValueError("tool_calls outside assistant")
        for ci, call in enumerate(calls):
            if not isinstance(call, dict):
                raise ValueError(f"Call {mi}:{ci} must be an object")
            if call.get("type") != "function":
                raise ValueError("Unsupported call type")
            fn = call["function"]
            args = json_value(fn["arguments"])
            if not isinstance(args, dict):
                raise ValueError("function.arguments must decode once to an object")
            fn["arguments"] = args
            schema = schemas.get(fn["name"])
            errors = (
                ["Unknown tool"]
                if schema is None
                else [e.message for e in validators.validator_for(schema)(schema).iter_errors(args)]
            )
            checks.append(
                {
                    "kind": "arguments",
                    "status": "invalid" if errors else "valid",
                    "message_index": mi,
                    "call_index": ci,
                    "errors": errors,
                }
            )
            call_id = call.get("id") or f"position:{mi}:{ci}"
            if call_id in seen:
                raise ValueError("Duplicate call id")
            seen.add(call_id)
            pending.append(call_id)
    if pending:
        # Only explicit mini-SWE terminal submissions may lack a final observation.
        calls = messages[-1].get("tool_calls") or []
        commands = [c["function"]["arguments"].get("command") for c in calls]
        if messages[-1]["role"] != "assistant" or not all(terminal_candidate(c) for c in commands):
            raise ValueError("Missing final observation for nonterminal action")
        checks.append(
            {
                "kind": "terminal_observation",
                "status": "unverified",
                "reason": "Terminal submission has no observation",
                "call_ids": pending,
            }
        )
    return messages, tools, checks


def serialize(messages, tools, tokenizer):
    """Locate actual calls via marker renders, never search reasoning/observations for tags."""
    kwargs = {"tools": tools, "tokenize": False, "add_generation_prompt": False}
    text = tokenizer.apply_chat_template(messages, **kwargs)
    marked = copy.deepcopy(messages)
    markers = []
    for mi, message in enumerate(marked):
        for ci, call in enumerate(message.get("tool_calls") or []):
            fn = call["function"]
            # Marker strings cannot occur in the source or rendered text.
            prefix = f"__ACTION_FOCUSED_SFT_{mi}_{ci}_BEGIN__"
            suffix = f"__ACTION_FOCUSED_SFT_{mi}_{ci}_END__"
            if prefix in text or suffix in text:
                raise ValueError("Action marker collision")
            args = fn["arguments"]
            # Qwen serializes mappings as JSON; only string arguments can be marked losslessly.
            if not args or not isinstance(next(reversed(args.values())), str):
                markers.append((mi, ci, None, None))
                continue
            fn["name"] = prefix + fn["name"]
            last = next(reversed(args))
            args[last] += suffix
            markers.append((mi, ci, prefix, suffix))
    marked_text = tokenizer.apply_chat_template(marked, **kwargs)
    candidates = []
    for mi, ci, prefix, suffix in markers:
        if prefix is None:
            candidates.append(
                {
                    "message_index": mi,
                    "call_index": ci,
                    "char_span": None,
                    "status": "unverified_nonstring_argument",
                }
            )
            continue
        name_pos = marked_text.index(prefix)
        start = marked_text.rfind("<tool_call>", 0, name_pos)
        marked_text = marked_text.replace(prefix, "", 1)
        arg_end = marked_text.index(suffix)
        marked_text = marked_text.replace(suffix, "", 1)
        end = marked_text.index("</tool_call>", arg_end) + len("</tool_call>")
        if start < 0:
            raise ValueError("Official template lacks expected tool-call envelope")
        candidates.append(
            {
                "message_index": mi,
                "call_index": ci,
                "char_span": [start, end],
                "status": "candidate_only",
            }
        )
    if marked_text != text:
        raise ValueError("Action marker render changed official serialization")
    return {
        "text": text,
        "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
        "action_candidates": candidates,
    }


class QuotaError(ValueError):
    def __init__(self, category, train, validation, errors):
        super().__init__(
            f"Insufficient valid tasks in {category}: train={train}, validation={validation}"
        )
        self.errors = errors


def select(rows, load, train_quota=250, validation_quota=25, seed=42):
    """Round-robin repository groups; reserve whole tasks globally across categories."""
    from jinja2 import TemplateError
    from jsonschema.exceptions import SchemaError

    rng = random.Random(seed)
    selected = {"train": [], "validation": [], "errors": []}
    used = set()
    for category in CATEGORIES:
        groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["category"] == category:
                groups[row["repo"]][row["task_key"]].append(row)
        repositories = sorted(groups)
        rng.shuffle(repositories)
        queues = {}
        for repo in repositories:
            tasks = sorted(groups[repo])
            rng.shuffle(tasks)
            for task in tasks:
                attempts = sorted(groups[repo][task], key=lambda r: r["trajectory_id"])
                rng.shuffle(attempts)
                groups[repo][task] = attempts
            queues[repo] = deque(tasks)
        counts = Counter()
        while repositories and sum(counts.values()) < train_quota + validation_quota:
            for repo in list(repositories):
                queue = queues[repo]
                while queue and queue[0] in used:
                    queue.popleft()
                if not queue:
                    repositories.remove(repo)
                    continue
                task = queue.popleft()
                for row in groups[repo][task]:
                    try:
                        record = load(row)
                    except (ValueError, KeyError, TypeError, SchemaError, TemplateError) as error:
                        selected["errors"].append({**row, "reason": str(error)})
                        continue
                    split = "validation" if counts["validation"] < validation_quota else "train"
                    selected[split].append(record)
                    counts[split] += 1
                    used.add(task)
                    break
                if sum(counts.values()) == train_quota + validation_quota:
                    break
        if counts["train"] != train_quota or counts["validation"] != validation_quota:
            raise QuotaError(category, counts["train"], counts["validation"], selected["errors"])
    return selected


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def benchmark_keys(config, offline):
    """Read all base AND style tasks, verifying against the original audit hashes."""
    from huggingface_hub import snapshot_download

    location = ROOT / "data/raw/omnicode_keys_source"
    if not offline:
        snapshot_download(
            config["leakage"]["benchmark_id"],
            repo_type="dataset",
            revision=config["leakage"]["benchmark_revision"],
            allow_patterns=["omnicode*.json"],
            local_dir=location,
        )
    expected = json.loads((ROOT / config["leakage"]["source_checksums"]).read_text())
    keys, counts, hashes = set(), {}, {}
    for style in (False, True):
        for language in ("cpp", "java", "python"):
            name = f"omnicode_{'style_' if style else ''}instances_{language}.json"
            path = location / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected[name]:
                raise ValueError(f"OmniCode audit checksum mismatch: {name}")
            hashes[name] = actual
            records = json.loads(path.read_text())
            if isinstance(records, dict):
                records = list(records.values())
            counts[name] = len(records)
            for record in records:
                instance = record.get("original_instance_id") or record["instance_id"]
                keys.add(task_key(record["repo"], instance, record.get("pull_number")))
    if sum(counts.values()) != 909:
        raise ValueError(f"Incomplete OmniCode task list: {counts}")
    return keys, {"counts": counts, "sha256": hashes, "unique_tasks": len(keys)}


class Source:
    """Cache only requested columns. Reference/model patches are never read."""

    def __init__(self, config, offline=False):
        from huggingface_hub import HfFileSystem

        self.config = config
        self.offline = offline
        self.fs = HfFileSystem()
        self.cache = ROOT / config["local_paths"]["raw"] / config["dataset"]["revision"]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.prefetch = defaultdict(list)

    def plan_prefetch(self, rows):
        for row in rows:
            self.prefetch[(row["shard"], row["row_group"])].append(row)

    def warm(self):
        representatives = [rows[0] for rows in self.prefetch.values()]
        # Two projected row groups at a time bound RAM while overlapping network reads.
        with ThreadPoolExecutor(max_workers=2) as executor:
            for row, _ in zip(
                representatives, executor.map(self.load, representatives), strict=True
            ):
                print(f"Cached candidate conversations: {row['shard']}", flush=True)

    def row_path(self, row):
        return self.cache / "rows" / f"{row['trajectory_id']}.parquet"

    def parquet(self, shard):
        import pyarrow.parquet as pq

        ds = self.config["dataset"]
        path = f"datasets/{ds['id']}@{ds['revision']}/{shard}"
        return pq.ParquetFile(self.fs.open(path, "rb"))

    def index_shard(self, shard):
        path = self.cache / (shard.replace("/", "__") + ".index.json")
        if path.exists():
            return json.loads(path.read_text())
        if self.offline:
            raise FileNotFoundError(path)
        parquet = self.parquet(shard)
        rows = []
        columns = [
            "repo",
            "instance_id",
            "trajectory_id",
            "language",
            "license",
            "resolved",
            "hf_dataset_name",
            "metadata.category",
            "metadata.teacher_model",
        ]
        for group in range(parquet.num_row_groups):
            for offset, row in enumerate(
                parquet.read_row_group(group, columns=columns).to_pylist()
            ):
                metadata = row.pop("metadata")
                row.update(
                    {
                        "category": metadata["category"],
                        "teacher_model": metadata["teacher_model"],
                        "shard": shard,
                        "row_group": group,
                        "row_offset": offset,
                    }
                )
                rows.append(row)
        write_json(path, rows)
        print(f"Indexed {shard}: {len(rows)}", flush=True)
        return rows

    def index(self):
        from fnmatch import fnmatch

        from huggingface_hub import HfApi

        listing = self.cache / "shards.json"
        if listing.exists():
            shards = json.loads(listing.read_text())
        elif self.offline:
            raise FileNotFoundError(listing)
        else:
            ds = self.config["dataset"]
            shards = sorted(
                s
                for s in HfApi().list_repo_files(
                    ds["id"], repo_type="dataset", revision=ds["revision"]
                )
                if fnmatch(s, ds["data_files"])
            )
            if len(shards) != 44:
                raise ValueError(f"Expected 44 audited shards; got {len(shards)}")
            write_json(listing, shards)
        with ThreadPoolExecutor(max_workers=4) as executor:
            batches = list(executor.map(self.index_shard, shards))
        return [row for batch in batches for row in batch]

    def group(self, shard, group):
        import pyarrow.parquet as pq

        path = self.cache / (shard.replace("/", "__") + f".group{group}.parquet")
        if path.exists():
            return pq.read_table(path)
        if self.offline:
            raise FileNotFoundError(path)
        table = self.parquet(shard).read_row_group(
            group, columns=["trajectory_id", "messages", "tools"]
        )
        temporary = path.with_suffix(".partial")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
        return table

    def load(self, row):
        import pyarrow.parquet as pq

        row_path = self.row_path(row)
        if not row_path.exists():
            table = self.group(row["shard"], row["row_group"])
            candidates = self.prefetch.pop((row["shard"], row["row_group"]), [])
            for candidate in candidates + [row]:
                path = self.row_path(candidate)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".partial")
                pq.write_table(
                    table.slice(candidate["row_offset"], 1), temporary, compression="zstd"
                )
                temporary.replace(path)
        record = pq.read_table(row_path).to_pylist()[0]
        if record["trajectory_id"] != row["trajectory_id"]:
            raise ValueError("Source row locator mismatch")
        return record


def length_summary(records):
    import numpy as np

    lengths = [r["token_count"] for r in records]
    return {
        "min": min(lengths),
        "max": max(lengths),
        "total": sum(lengths),
        "quantiles": {str(q): float(np.quantile(lengths, q)) for q in (0.5, 0.9, 0.95, 0.99)},
        "above": {
            str(n): sum(v > n for v in lengths) for n in (8192, 16384, 32768, 65536, 131072, 262144)
        },
    }


def operation_type(command):
    """Descriptive first-program heuristic; never a command-availability verdict."""
    if not isinstance(command, str):
        return "invalid_nonstring"
    if terminal_candidate(command):
        return "submit"
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "dynamic_or_other"
    if not tokens:
        return "dynamic_or_other"
    program = tokens[0]
    groups = {
        "test": {"pytest", "unittest", "tox", "ctest"},
        "git": {"git"},
        "read_search": {"cat", "ls", "find", "rg", "grep", "head", "tail", "pwd", "wc"},
        "edit": {"sed", "perl", "apply_patch", "patch", "tee"},
        "build_package": {"make", "cmake", "pip", "npm", "yarn", "cargo", "mvn", "gradle"},
        "script_interpreter": {"python", "python3", "bash", "sh", "node"},
    }
    return next(
        (name for name, programs in groups.items() if program in programs), "dynamic_or_other"
    )


def partition_pool(rows, assignment, selected_ids, benchmark):
    return [
        {
            **row,
            "partition": assignment.get(row["task_key"], "phase2_candidate"),
            "selected_phase1": row["trajectory_id"] in selected_ids,
        }
        for row in rows
        if row["task_key"] not in benchmark
    ]


def split_names(phase):
    return (
        ("train", "validation", "test") if "test_per_category" in phase else ("train", "validation")
    )


def split_heldout(result, phase):
    """Split the original held-out tasks without changing train selection."""
    if not isinstance(phase["validation_per_category"], dict):
        return result
    output = {**result, "validation": []}
    if "test_per_category" in phase:
        output["test"] = []
    rng = random.Random(phase["seed"])
    for category in CATEGORIES:
        rows = sorted(
            (r for r in result["validation"] if r["category"] == category),
            key=lambda r: r["task_key"],
        )
        n_valid = phase["validation_per_category"][category]
        n_test = (
            phase["test_per_category"][category]
            if "test_per_category" in phase
            else phase["heldout_per_category"] - n_valid
        )
        if len(rows) != n_valid + n_test:
            raise ValueError(f"Wrong held-out quota for {category}")
        rng.shuffle(rows)
        output["validation"].extend(rows[:n_valid])
        if "test" in output:
            output["test"].extend(rows[n_valid:])
    return output


def verify_selection(result, benchmark, config):
    phase = config["phase1"]
    keys = {}
    for split in split_names(phase):
        quota = phase[f"{split}_per_category"]
        expected = quota if isinstance(quota, dict) else dict.fromkeys(CATEGORIES, quota)
        records = result[split]
        counts = Counter(r["category"] for r in records)
        if counts != Counter(expected):
            raise ValueError(f"Wrong quotas: {split} {counts}")
        keys[split] = {r["task_key"] for r in records}
        if len(keys[split]) != len(records) or keys[split] & benchmark:
            raise ValueError(f"Duplicate or leaked tasks: {split}")
        if any(keys[split] & previous for name, previous in keys.items() if name != split):
            raise ValueError("Cross-split task leakage")


def verify_hashes(directory, hashes):
    for name, expected in hashes.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")


def verify_outputs(config_path):
    """Independent offline verification against cached source and official tokenizer."""
    from transformers import AutoTokenizer

    config = json.loads(config_path.read_text())
    out = ROOT / config["local_paths"]["processed"]
    report = json.loads((out / "report.json").read_text())
    manifest = json.loads((out / "task_selection_manifest.json").read_text())
    if manifest["config"] != config:
        raise ValueError("Manifest config mismatch")
    verify_hashes(out, report["artifact_sha256"])
    benchmark, _ = benchmark_keys(config, True)
    tokenizer_path = ROOT / config["student"]["tokenizer_path"]
    verify_hashes(tokenizer_path, manifest["tokenizer"]["file_sha256"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if digest(tokenizer.get_chat_template()) != manifest["tokenizer"]["template_sha256"]:
        raise ValueError("Template checksum mismatch")
    source = Source(config, True)
    result = {
        split: [json.loads(line) for line in (out / f"{split}.jsonl").open()]
        for split in split_names(config["phase1"])
    }
    verify_selection(result, benchmark, config)
    assignments = {r["task_key"]: split for split in result for r in result[split]}
    selected_ids = {r["trajectory_id"] for split in result for r in result[split]}
    pool_counts = Counter()
    with (out / "task_pool.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["task_key"] in benchmark:
                raise ValueError("Benchmark task in phase2 pool")
            if row["partition"] != assignments.get(row["task_key"], "phase2_candidate"):
                raise ValueError("Task attempts split across partitions")
            if row["selected_phase1"] != (row["trajectory_id"] in selected_ids):
                raise ValueError("Incorrect selected_phase1 flag")
            pool_counts[row["partition"]] += 1
    summary = {"pool_attempts": dict(pool_counts), "splits": {}}
    for split, records in result.items():
        operations = Counter()
        for record in records:
            raw = source.load(record)
            messages, tools, checks = normalize(raw)
            if record["messages"] != messages or record["tools"] != tools:
                raise ValueError("Normalized conversation differs from source")
            if record["checks"] != checks:
                raise ValueError("Schema/sequence checks changed")
            rendered = serialize(messages, tools, tokenizer)
            if any(record[key] != rendered[key] for key in rendered):
                raise ValueError("Official serialization/token count/action spans differ")
            for message in messages:
                for call in message.get("tool_calls") or []:
                    operations[operation_type(call["function"]["arguments"].get("command"))] += 1
        summary["splits"][split] = {
            "records_verified_against_source": len(records),
            "operation_type_first_program_heuristic": dict(operations),
        }
    summary["verified"] = True
    summary["commands_executed"] = 0
    write_json(out / "verification_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run(config_path, offline=False):
    """Write outputs only after complete quota and deterministic selection checks."""
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    config = json.loads(config_path.read_text())
    phase = config["phase1"]
    out = ROOT / config["local_paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        raise FileExistsError(
            f"Completed output already exists: {out}; verify or choose a new output directory"
        )
    benchmark, benchmark_info = benchmark_keys(config, offline)
    tokenizer_path = ROOT / config["student"]["tokenizer_path"]
    revision = config["student"]["revision"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Student tokenizer must be pinned to a resolved commit SHA")
    if not offline:
        snapshot_download(
            config["student"]["primary"],
            revision=revision,
            allow_patterns=[
                "tokenizer*",
                "chat_template*",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
                "config.json",
            ],
            local_dir=tokenizer_path,
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    template = tokenizer.get_chat_template()
    source = Source(config, offline)
    all_rows = source.index()
    eligible, excluded, keyed_rows = [], [], []
    for source_row in all_rows:
        row = dict(source_row)
        try:
            row["task_key"] = task_key(row["repo"], row["instance_id"])
        except ValueError as error:
            excluded.append({**row, "reason": str(error)})
            continue
        teacher = row["teacher_model"]
        keyed_rows.append(row)
        reason = None
        if row["task_key"] in benchmark:
            reason = "OmniCode task overlap"
        elif row["resolved"] != 1:
            reason = "resolved != 1"
        elif (
            teacher.get("name") != config["selection"]["teacher"]
            or teacher.get("enable_thinking") is not True
        ):
            reason = "Teacher/Thinking filter"
        elif row["category"] not in CATEGORIES:
            reason = "Unsupported category"
        if reason:
            excluded.append({**row, "reason": reason})
        else:
            row["repo"] = row["repo"].lower()
            eligible.append(row)
    write_jsonl(out / "excluded.jsonl", excluded)
    # Anticipate selection plus reserve candidates, so each large shard is decoded once normally.
    try:
        planned = select(
            eligible,
            lambda row: row,
            train_quota=phase["train_per_category"] + 100,
            validation_quota=phase.get("heldout_per_category", phase["validation_per_category"]),
            seed=phase["seed"],
        )
    except QuotaError:
        planned = select(
            eligible,
            lambda row: row,
            train_quota=phase["train_per_category"],
            validation_quota=phase.get("heldout_per_category", phase["validation_per_category"]),
            seed=phase["seed"],
        )
    source.plan_prefetch(planned["train"] + planned["validation"])
    source.warm()
    prepared = {}

    def load(row):
        tid = row["trajectory_id"]
        if tid not in prepared:
            raw = source.load(row)
            messages, tools, checks = normalize(raw)
            rendered = serialize(messages, tools, tokenizer)
            prepared[tid] = {
                **row,
                "messages": messages,
                "tools": tools,
                "checks": checks,
                **rendered,
                "source_messages_sha256": digest(json.dumps(raw["messages"], ensure_ascii=False)),
                "command_availability": "unverified_no_harness_environment",
            }
            if len(prepared) % 25 == 0:
                print(f"Prepared {len(prepared)} trajectories", flush=True)
        return prepared[tid]

    options = {
        "train_quota": phase["train_per_category"],
        "validation_quota": phase.get("heldout_per_category", phase["validation_per_category"]),
        "seed": phase["seed"],
    }
    try:
        result = split_heldout(select(eligible, load, **options), phase)
    except QuotaError as error:
        write_jsonl(out / "errors.jsonl", error.errors)
        write_json(out / "failure.json", {"reason": str(error), "config": config})
        raise
    verify_selection(result, benchmark, config)
    repeat = split_heldout(select(list(reversed(eligible)), load, **options), phase)
    for split in split_names(config["phase1"]):
        if [r["trajectory_id"] for r in result[split]] != [
            r["trajectory_id"] for r in repeat[split]
        ]:
            raise ValueError("Selection is not deterministic")
    selected = [r for split in split_names(phase) for r in result[split]]
    assignment = {
        r["task_key"]: split for split in split_names(config["phase1"]) for r in result[split]
    }
    # ALL attempts, including unsuccessful attempts, retain the selected task's partition.
    pool = partition_pool(keyed_rows, assignment, {r["trajectory_id"] for r in selected}, benchmark)
    write_jsonl(out / "task_pool.jsonl", pool)
    for split in split_names(config["phase1"]):
        write_jsonl(out / f"{split}.jsonl", result[split])
    write_jsonl(out / "errors.jsonl", result["errors"])
    write_jsonl(
        out / "checks.jsonl",
        [
            {"trajectory_id": r["trajectory_id"], **check}
            for r in selected
            for check in r["checks"]
            if check["status"] != "valid"
        ],
    )
    provenance_fields = (
        "trajectory_id",
        "task_key",
        "repo",
        "instance_id",
        "category",
        "language",
        "shard",
        "row_group",
        "row_offset",
        "hf_dataset_name",
        "teacher_model",
    )
    manifest = {
        "config": config,
        "config_sha256": digest(config_path.read_text()),
        "benchmark": benchmark_info,
        "tokenizer": {
            "id": config["student"]["primary"],
            "revision": revision,
            "template_sha256": digest(template),
            "transformers_version": transformers.__version__,
            "file_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in tokenizer_path.iterdir()
                if p.is_file()
            },
            "add_generation_prompt": False,
        },
        "selection": {
            split: [{k: r.get(k) for k in provenance_fields} for r in result[split]]
            for split in split_names(config["phase1"])
        },
        "phase2_rule": (
            "Exclude ALL held-out and benchmark task keys "
            "regardless of attempt/category/resolved status"
        ),
    }
    write_json(out / "task_selection_manifest.json", manifest)
    report = {
        "source_rows": len(all_rows),
        "eligible_rows": len(eligible),
        "excluded_rows": len(excluded),
        "invalid_candidate_rows": len(result["errors"]),
        "deterministic_rerun": True,
        "splits": {
            split: {
                "trajectories": len(result[split]),
                "length_tokens": length_summary(result[split]),
                "coverage": {
                    field: dict(Counter(r[field] for r in result[split]))
                    for field in ("category", "repo", "language", "hf_dataset_name")
                },
                "operation_type_first_program_heuristic": dict(
                    Counter(
                        operation_type(call["function"]["arguments"].get("command"))
                        for row in result[split]
                        for message in row["messages"]
                        for call in message.get("tool_calls") or []
                    )
                ),
            }
            for split in split_names(config["phase1"])
        },
        "tool_argument_checks": dict(
            Counter(c["status"] for r in selected for c in r["checks"] if c["kind"] == "arguments")
        ),
        "commands_executed": 0,
        "student_metrics_measured": False,
        "limitations": [
            "Observation links without IDs are positional and unverified",
            "Action spans are candidates, not labels or proof of valid commands",
            "No truncation, packing, training labels, training or student evaluation",
        ],
    }
    report["artifact_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir()
        if p.is_file() and p.name not in {"report.json", "verification_report.json", "failure.json"}
    }
    write_json(out / "report.json", report)
    print(
        json.dumps(
            {
                split: report["splits"][split]["trajectories"]
                for split in split_names(config["phase1"])
            }
        )
    )


def split_existing_heldout(config_path):
    """Migrate existing 100 held-out records; never regenerate or rewrite train."""
    config = json.loads(config_path.read_text())
    if "test_per_category" not in config["phase1"]:
        raise ValueError(
            "Legacy split migration requires internal test quotas; current test is external"
        )
    out = ROOT / config["local_paths"]["processed"]
    manifest = json.loads((out / "task_selection_manifest.json").read_text())
    report = json.loads((out / "report.json").read_text())
    if "test" in manifest["selection"] or (out / "test.jsonl").exists():
        raise ValueError("Test split already exists; refusing to repartition it")
    verify_hashes(out, report["artifact_sha256"])
    old_phase = manifest["config"]["phase1"]
    expected_config = copy.deepcopy(manifest["config"])
    expected_config["phase1"] = config["phase1"]
    if expected_config != config:
        raise ValueError("Migration may only change phase1 split configuration")
    if config["phase1"]["heldout_per_category"] != old_phase["validation_per_category"]:
        raise ValueError("Original held-out quota changed")
    result = {
        split: [json.loads(line) for line in (out / f"{split}.jsonl").open()]
        for split in ("train", "validation")
    }
    original = {r["task_key"]: r for r in result["validation"]}
    result = split_heldout(result, config["phase1"])
    benchmark, _ = benchmark_keys(config, True)
    verify_selection(result, benchmark, config)
    repeat = split_heldout(
        {"train": result["train"], "validation": list(reversed(list(original.values())))},
        config["phase1"],
    )
    for split in ("validation", "test"):
        if result[split] != repeat[split]:
            raise ValueError("Held-out split is not deterministic")
    assignment = {r["task_key"]: split for split in result for r in result[split]}
    pool = [json.loads(line) for line in (out / "task_pool.jsonl").open()]
    for row in pool:
        row["partition"] = assignment.get(row["task_key"], "phase2_candidate")
    old_selection = {r["task_key"]: r for r in manifest["selection"]["validation"]}
    for split in ("validation", "test"):
        records = result[split]
        manifest["selection"][split] = [old_selection[r["task_key"]] for r in records]
        report["splits"][split] = {
            "trajectories": len(records),
            "length_tokens": length_summary(records),
            "coverage": {
                field: dict(Counter(r[field] for r in records))
                for field in ("category", "repo", "language", "hf_dataset_name")
            },
            "operation_type_first_program_heuristic": dict(
                Counter(
                    operation_type(call["function"]["arguments"].get("command"))
                    for r in records
                    for m in r["messages"]
                    for call in m.get("tool_calls") or []
                )
            ),
        }
    manifest["config"] = config
    manifest["config_sha256"] = digest(config_path.read_text())
    manifest["phase2_rule"] = (
        "Exclude ALL validation AND test task keys regardless of attempt/category/resolved status"
    )
    manifest["heldout_migration"] = {
        "original_validation_task_keys": sorted(original),
        "train_sha256_unchanged": report["artifact_sha256"]["train.jsonl"],
        "seed": config["phase1"]["seed"],
    }
    # All preconditions are checked before modifying any artifacts.
    for split in ("validation", "test"):
        write_jsonl(out / f"{split}.jsonl", result[split])
    write_jsonl(out / "task_pool.jsonl", pool)
    write_json(out / "task_selection_manifest.json", manifest)
    report["artifact_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir()
        if p.is_file() and p.name not in {"report.json", "verification_report.json", "failure.json"}
    }
    write_json(out / "report.json", report)
    (out / "verification_report.json").unlink(missing_ok=True)
    print(json.dumps({split: len(result[split]) for split in result}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/datasets/nvidia_mini_swe_agent_v1_2_action_focused_sft.json",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use cached source columns and tokenizer only"
    )
    parser.add_argument("--verify", action="store_true", help="Verify completed outputs offline")
    parser.add_argument("--split-existing-heldout", action="store_true")
    args = parser.parse_args()
    if args.split_existing_heldout:
        split_existing_heldout(args.config)
    elif args.verify:
        verify_outputs(args.config)
    else:
        run(args.config, offline=args.offline)


if __name__ == "__main__":
    main()
