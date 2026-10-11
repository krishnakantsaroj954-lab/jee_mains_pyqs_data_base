#!/usr/bin/env python3
"""Decode a pinned pickle into inert Python containers without executing pickle globals.

This is a limited static interpreter for the opcode subset used by the JEE v007
cache. It NEVER calls pickle.load/loads, importlib, eval, or any decoded callable.
Unknown opcodes fail closed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickletools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MARK = object()


@dataclass(eq=False)
class GlobalRef:
    module: str
    name: str


@dataclass(eq=False)
class StaticObject:
    cls: Any
    args: Any
    kwargs: Any = None
    state: Any = None


@dataclass(eq=False)
class StaticReduce:
    callable: Any
    args: Any
    state: Any = None


@dataclass(eq=False)
class StaticExtension:
    code: Any


@dataclass(eq=False)
class StaticPersistent:
    value: Any


def _marked_values(stack: list[Any]) -> list[Any]:
    for index in range(len(stack) - 1, -1, -1):
        if stack[index] is MARK:
            values = stack[index + 1:]
            del stack[index:]
            return values
    raise ValueError("pickle MARK not found")


def _dict_from_pairs(values: list[Any]) -> dict[Any, Any]:
    if len(values) % 2:
        raise ValueError("odd number of key/value items in static DICT")
    out: dict[Any, Any] = {}
    for i in range(0, len(values), 2):
        key, value = values[i], values[i + 1]
        try:
            out[key] = value
        except TypeError as exc:
            raise ValueError("unhashable static dictionary key; refusing lossy decode") from exc
    return out


def decode_inert(data: bytes) -> Any:
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    operations = 0

    def push(value: Any) -> None:
        stack.append(value)
        if len(stack) > 2_000_000:
            raise ValueError("stack limit exceeded; refusing oversized pickle structure")

    for op, arg, _pos in pickletools.genops(data):
        name = op.name
        operations += 1
        if operations > 5_000_000:
            raise ValueError("opcode limit exceeded")
        if name in {"PROTO", "FRAME"}:
            continue
        if name == "STOP":
            if not stack:
                raise ValueError("STOP with empty stack")
            return stack[-1]
        if name == "MARK":
            push(MARK)
        elif name == "POP":
            stack.pop()
        elif name == "POP_MARK":
            _marked_values(stack)
        elif name == "DUP":
            push(stack[-1])
        elif name in {
            "NONE", "NEWTRUE", "NEWFALSE", "INT", "BININT", "BININT1",
            "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT",
            "STRING", "BINSTRING", "SHORT_BINSTRING", "BINUNICODE",
            "SHORT_BINUNICODE", "BINUNICODE8", "UNICODE", "BINBYTES",
            "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8",
        }:
            if name == "NONE":
                value = None
            elif name == "NEWTRUE":
                value = True
            elif name == "NEWFALSE":
                value = False
            else:
                value = arg
            push(value)
        elif name == "EMPTY_DICT":
            push({})
        elif name == "EMPTY_LIST":
            push([])
        elif name == "EMPTY_TUPLE":
            push(())
        elif name == "EMPTY_SET":
            push(set())
        elif name == "LIST":
            push(list(_marked_values(stack)))
        elif name == "TUPLE":
            push(tuple(_marked_values(stack)))
        elif name == "DICT":
            push(_dict_from_pairs(_marked_values(stack)))
        elif name == "TUPLE1":
            a = stack.pop()
            push((a,))
        elif name == "TUPLE2":
            b, a = stack.pop(), stack.pop()
            push((a, b))
        elif name == "TUPLE3":
            c, b, a = stack.pop(), stack.pop(), stack.pop()
            push((a, b, c))
        elif name == "APPEND":
            value, target = stack.pop(), stack[-1]
            if not isinstance(target, list):
                raise ValueError("APPEND target is not a list")
            target.append(value)
        elif name == "APPENDS":
            values = _marked_values(stack)
            if not stack or not isinstance(stack[-1], list):
                raise ValueError("APPENDS target is not a list")
            stack[-1].extend(values)
        elif name == "SETITEM":
            value, key = stack.pop(), stack.pop()
            target = stack[-1]
            if not isinstance(target, dict):
                raise ValueError("SETITEM target is not a dict")
            try:
                target[key] = value
            except TypeError as exc:
                raise ValueError("unhashable SETITEM key") from exc
        elif name == "SETITEMS":
            values = _marked_values(stack)
            if not stack or not isinstance(stack[-1], dict):
                raise ValueError("SETITEMS target is not a dict")
            target = stack[-1]
            if len(values) % 2:
                raise ValueError("odd number of SETITEMS values")
            for i in range(0, len(values), 2):
                try:
                    target[values[i]] = values[i + 1]
                except TypeError as exc:
                    raise ValueError("unhashable SETITEMS key") from exc
        elif name == "ADDITEMS":
            values = _marked_values(stack)
            if not stack or not isinstance(stack[-1], set):
                raise ValueError("ADDITEMS target is not a set")
            for value in values:
                try:
                    stack[-1].add(value)
                except TypeError as exc:
                    raise ValueError("unhashable ADDITEMS value") from exc
        elif name == "FROZENSET":
            values = _marked_values(stack)
            try:
                push(frozenset(values))
            except TypeError:
                # Preserve content without executing user-defined hashing.
                push(("STATIC_FROZENSET", tuple(values)))
        elif name in {"GET", "BINGET", "LONG_BINGET"}:
            push(memo[int(arg)])
        elif name in {"PUT", "BINPUT", "LONG_BINPUT"}:
            memo[int(arg)] = stack[-1]
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name == "GLOBAL":
            module, cls_name = str(arg).split(" ", 1)
            push(GlobalRef(module, cls_name))
        elif name == "STACK_GLOBAL":
            cls_name, module = stack.pop(), stack.pop()
            if not isinstance(module, str) or not isinstance(cls_name, str):
                raise ValueError("STACK_GLOBAL operands are not strings")
            push(GlobalRef(module, cls_name))
        elif name == "NEWOBJ":
            args, cls = stack.pop(), stack.pop()
            push(StaticObject(cls, args))
        elif name == "NEWOBJ_EX":
            kwargs, args, cls = stack.pop(), stack.pop(), stack.pop()
            push(StaticObject(cls, args, kwargs))
        elif name == "OBJ":
            values = _marked_values(stack)
            if not values:
                raise ValueError("OBJ has no class")
            push(StaticObject(values[0], tuple(values[1:])))
        elif name == "INST":
            values = _marked_values(stack)
            module, cls_name = str(arg).split(" ", 1)
            push(StaticObject(GlobalRef(module, cls_name), tuple(values)))
        elif name == "BUILD":
            state = stack.pop()
            if not stack or not isinstance(stack[-1], (StaticObject, StaticReduce)):
                raise ValueError("BUILD target is not an inert object")
            stack[-1].state = state
        elif name == "REDUCE":
            args, function = stack.pop(), stack.pop()
            # Deliberately retain the callable as inert data; never invoke it.
            push(StaticReduce(function, args))
        elif name in {"EXT1", "EXT2", "EXT4"}:
            push(StaticExtension(arg))
        elif name == "PERSID":
            push(StaticPersistent(arg))
        elif name == "BINPERSID":
            push(StaticPersistent(stack.pop()))
        elif name == "NEXT_BUFFER":
            push(("STATIC_OUT_OF_BAND_BUFFER",))
        elif name == "READONLY_BUFFER":
            if not stack:
                raise ValueError("READONLY_BUFFER with empty stack")
        else:
            raise ValueError(f"unsupported opcode {name}; no executable fallback is allowed")

    raise ValueError("pickle stream ended without STOP")


def class_name(obj: Any) -> str:
    cls = obj.cls if isinstance(obj, (StaticObject, StaticReduce)) else obj
    if isinstance(cls, GlobalRef):
        return f"{cls.module}.{cls.name}"
    return type(cls).__name__


def object_fields(obj: StaticObject | StaticReduce) -> dict[str, Any]:
    state = obj.state
    if isinstance(state, dict):
        result: dict[str, Any] = {}
        inner = state.get("__dict__")
        if isinstance(inner, dict):
            result.update(inner)
        for key, value in state.items():
            if key not in {"__dict__", "__slots__"} and isinstance(key, str):
                result[key] = value
        slots = state.get("__slots__")
        if isinstance(slots, dict):
            result.update({k: v for k, v in slots.items() if isinstance(k, str)})
        return result
    if isinstance(state, tuple):
        result = {}
        for part in state:
            if isinstance(part, dict):
                result.update({k: v for k, v in part.items() if isinstance(k, str)})
        return result
    return {}


def walk_graph(root: Any):
    pending = [root]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            continue
        ident = id(value)
        if ident in seen:
            continue
        seen.add(ident)
        yield value
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        elif isinstance(value, StaticObject):
            pending.append(value.args)
            pending.append(value.kwargs)
            pending.append(value.state)
        elif isinstance(value, StaticReduce):
            pending.extend((value.callable, value.args, value.state))
        elif isinstance(value, StaticPersistent):
            pending.append(value.value)


def strings_contain_img(value: Any, visited: set[int] | None = None) -> bool:
    if isinstance(value, str):
        return "<img" in value.lower()
    if value is None or isinstance(value, (bytes, int, float, bool)):
        return False
    visited = visited or set()
    ident = id(value)
    if ident in visited:
        return False
    visited.add(ident)
    if isinstance(value, dict):
        return any(strings_contain_img(k, visited) or strings_contain_img(v, visited)
                   for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(strings_contain_img(v, visited) for v in value)
    if isinstance(value, (StaticObject, StaticReduce)):
        return strings_contain_img(value.state, visited)
    return False


def build_report(data: bytes, root: Any) -> dict[str, Any]:
    all_nodes = list(walk_graph(root))
    objects = [node for node in all_nodes if isinstance(node, StaticObject)]
    class_counts = collections.Counter(class_name(node) for node in objects)

    candidates: list[tuple[StaticObject, dict[str, Any]]] = []
    for obj in objects:
        fields = object_fields(obj)
        if "question_id" in fields and "subject" in fields and (
            "question" in fields or "options" in fields
        ):
            candidates.append((obj, fields))

    def present(fields: dict[str, Any], key: str) -> bool:
        value = fields.get(key)
        return value is not None and value != "" and value != [] and value != {}

    subjects = collections.Counter(str(fields.get("subject", "MISSING")) for _, fields in candidates)
    years = collections.Counter(str(fields.get("year", "MISSING")) for _, fields in candidates)
    question_types = collections.Counter(str(fields.get("type", "MISSING")) for _, fields in candidates)
    field_names = (
        "question_id", "examGroup", "exam", "subject", "chapter", "year",
        "paperTitle", "difficulty", "topic", "type", "examDate", "answer",
        "question", "options", "correct_options", "explanation",
        "isOutOfSyllabus", "isBonus",
    )
    field_coverage = {
        key: {
            "present_nonempty": sum(present(fields, key) for _, fields in candidates),
            "total_candidates": len(candidates),
        }
        for key in field_names
    }

    unique_ids = {
        str(fields.get("question_id"))
        for _, fields in candidates
        if present(fields, "question_id")
    }
    question_images = sum(strings_contain_img(fields.get("question")) for _, fields in candidates)
    option_images = sum(strings_contain_img(fields.get("options")) for _, fields in candidates)
    explanation_images = sum(strings_contain_img(fields.get("explanation")) for _, fields in candidates)

    root_kind = type(root).__name__
    root_length = len(root) if isinstance(root, (dict, list, tuple, set)) else None
    return {
        "status": "STATIC_DECODE_COMPLETE",
        "method": "pickletools opcode interpreter; no pickle.load/loads, eval, import, or callable execution",
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "source_bytes": len(data),
        "root_type": root_kind,
        "root_length": root_length,
        "static_node_count": len(all_nodes),
        "inert_object_count": len(objects),
        "inert_object_classes_top20": dict(class_counts.most_common(20)),
        "question_record_candidates": len(candidates),
        "distinct_nonempty_question_ids": len(unique_ids),
        "subject_counts": dict(sorted(subjects.items())),
        "year_counts": dict(sorted(years.items())),
        "question_type_counts": dict(sorted(question_types.items())),
        "field_coverage": field_coverage,
        "records_with_html_images": {
            "question_content": question_images,
            "options": option_images,
            "explanation": explanation_images,
        },
        "sample_metadata_only": [
            {
                "question_id": str(fields.get("question_id", ""))[:80],
                "subject": str(fields.get("subject", ""))[:50],
                "year": str(fields.get("year", ""))[:20],
                "chapter": str(fields.get("chapter", ""))[:100],
                "difficulty": str(fields.get("difficulty", ""))[:50],
                "type": str(fields.get("type", ""))[:50],
            }
            for _, fields in candidates[:3]
        ],
        "warnings": [
            "Candidate count is based on inert object fields and must be reviewed against cache semantics.",
            "No question text, options, answers, explanations, or image URLs are written to the report.",
            "Data licensing/redistribution rights remain unverified.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = args.input.read_bytes()
    root = decode_inert(data)
    report = build_report(data, root)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "source_sha256": report["source_sha256"],
        "source_bytes": report["source_bytes"],
        "root_type": report["root_type"],
        "root_length": report["root_length"],
        "inert_object_count": report["inert_object_count"],
        "inert_object_classes_top20": report["inert_object_classes_top20"],
        "question_record_candidates": report["question_record_candidates"],
        "distinct_nonempty_question_ids": report["distinct_nonempty_question_ids"],
        "subject_counts": report["subject_counts"],
        "year_counts": report["year_counts"],
        "field_coverage": report["field_coverage"],
        "records_with_html_images": report["records_with_html_images"],
        "report_path": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
