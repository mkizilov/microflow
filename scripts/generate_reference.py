"""Generate the CLI and HTTP API reference from the source. Standard library only.

The reference is read from the code, not written by hand: Typer commands and their options come from
the ``@...command`` decorators, and HTTP routes from the ``@...get``/``@...post`` decorators. Run it
after changing a command or route:

    python3 scripts/generate_reference.py

CI runs it and fails if the committed reference differs from the output, so the reference cannot
drift from the code.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "microflow"
OUT_DIR = ROOT / "docs" / "reference"
_PATH_PARAMETERS = {"job_id", "file_path", "manager", "body"}
COMMAND_FILES = {
    "cli.py": "root",
    "ml/cli.py": "ml",
    "stats/cli.py": "stats",
}


@dataclass
class Option:
    kind: str
    names: list[str]
    help: str
    default: str


@dataclass
class Command:
    group: str
    name: str
    doc: str
    options: list[Option] = field(default_factory=list)


@dataclass
class Route:
    method: str
    path: str
    protected: bool
    status: int
    body: str
    summary: str
    query: str


def _literal(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return repr(ast.literal_eval(node))
    except ValueError:
        return ast.unparse(node)


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    return " ".join(doc.strip().split("\n\n")[0].split())


def _typer_info(annotation: ast.AST) -> ast.Call | None:
    """The ``typer.Option(...)`` or ``typer.Argument(...)`` call inside an Annotated[...] hint."""
    if not isinstance(annotation, ast.Subscript) or ast.unparse(annotation.value) != "Annotated":
        return None
    elements = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else []
    for element in elements:
        if isinstance(element, ast.Call) and ast.unparse(element.func) in (
            "typer.Option",
            "typer.Argument",
        ):
            return element
    return None


def _commands(path: Path, group: str | None, apps: dict[str, str]) -> list[Command]:
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    found: list[Command] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "command"
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id in apps
            ):
                continue
            name = node.name.replace("_", "-")
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                name = str(decorator.args[0].value)
            command = Command(
                group=apps[decorator.func.value.id],
                name=name,
                doc=_first_paragraph(ast.get_docstring(node)),
            )
            defaults = dict(
                zip(
                    [arg.arg for arg in node.args.args][-len(node.args.defaults) :]
                    if node.args.defaults
                    else [],
                    node.args.defaults,
                    strict=True,
                )
            )
            for arg in node.args.args:
                info = _typer_info(arg.annotation) if arg.annotation else None
                if info is None:
                    continue
                is_argument = ast.unparse(info.func).endswith("Argument")
                kind = "argument" if is_argument else "option"
                positional = [
                    str(item.value)
                    for item in info.args
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                ]
                names = positional if kind == "option" and positional else [arg.arg]
                if kind == "option" and not positional:
                    names = ["--" + arg.arg.replace("_", "-")]
                help_text = _literal(_keyword(info, "help"))
                command.options.append(
                    Option(
                        kind=kind,
                        names=names,
                        help=help_text.strip("'\""),
                        default=_literal(defaults.get(arg.arg)),
                    )
                )
            found.append(command)
    del group
    return found


def _add_typer_groups(path: Path) -> dict[str, str]:
    """Map Typer app variable names to the group names they are mounted under."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_typer"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            name = _keyword(node, "name")
            if isinstance(name, ast.Constant):
                mapping[node.args[0].id] = str(name.value)
    return mapping


def cli_reference() -> str:
    groups = _add_typer_groups(SOURCE / "cli.py")
    apps: dict[str, str] = {"app": "root"}
    apps.update({name: group for name, group in groups.items()})
    commands: list[Command] = []
    for relative, group in COMMAND_FILES.items():
        file_apps = dict(apps) if group == "root" else {}
        if group != "root":
            module_app = {"ml": "ml_app", "stats": "stats_app"}[group]
            file_apps = {module_app: group}
        if group == "root":
            file_apps["plugins_app"] = "plugins"
        commands.extend(_commands(SOURCE / relative, group, file_apps))
    lines = [
        "# CLI reference",
        "",
        "<!-- Generated by scripts/generate_reference.py from the Typer decorators. "
        "Do not edit. -->",
        "",
        "Every command is `microflow [group] command [options] [arguments]`. "
        "Run `microflow --help`",
        "or `microflow <group> --help` for the same information in a terminal.",
        "",
    ]
    for group in sorted({command.group for command in commands}, key=_group_order):
        heading = "Top-level commands" if group == "root" else f"`microflow {group}` commands"
        lines += [f"## {heading}", ""]
        for command in commands:
            if command.group != group:
                continue
            prefix = "microflow " if group == "root" else f"microflow {group} "
            signature = " ".join(
                f"[{opt.names[0]}]" if opt.kind == "option" else f"<{opt.names[0]}>"
                for opt in command.options
                if opt.kind == "argument"
            )
            lines += [f"### `{prefix}{command.name}`", ""]
            if command.doc:
                lines += [command.doc, ""]
            usage = f"{prefix}{command.name} {signature}".rstrip()
            lines += [f"Usage: `{usage}`", ""]
            if command.options:
                lines += ["| Kind | Name | Default | Description |", "| --- | --- | --- | --- |"]
                for opt in command.options:
                    default = "" if opt.default in ("None", "") else f"`{opt.default}`"
                    names = ", ".join(f"`{name}`" for name in opt.names)
                    lines.append(f"| {opt.kind} | {names} | {default} | {opt.help} |")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _group_order(group: str) -> int:
    return {"root": 0, "ml": 1, "stats": 2, "plugins": 3}.get(group, 9)


def http_reference() -> str:
    path = SOURCE / "service" / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    routers: dict[str, bool] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == "APIRouter":
                protected = _keyword(node.value, "dependencies") is not None
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        routers[target.id] = protected
    routes: list[Route] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id in routers
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
            ):
                continue
            body = ""
            query: list[str] = []
            defaults = dict(
                zip(
                    [arg.arg for arg in node.args.args][-len(node.args.defaults) :]
                    if node.args.defaults
                    else [],
                    node.args.defaults,
                    strict=True,
                )
            )
            for arg in node.args.args:
                annotation = ast.unparse(arg.annotation) if arg.annotation is not None else ""
                if annotation.endswith("Request"):
                    body = annotation
                elif arg.arg not in _PATH_PARAMETERS and arg.arg != "request":
                    default = _literal(defaults.get(arg.arg))
                    text = f"`{arg.arg}`: {annotation or 'any'}"
                    query.append(text + (f" = {default}" if default else ""))
            status = _keyword(decorator, "status_code")
            routes.append(
                Route(
                    method=decorator.func.attr.upper(),
                    path="/v1" + str(decorator.args[0].value),
                    protected=routers[decorator.func.value.id],
                    status=int(status.value) if isinstance(status, ast.Constant) else 200,
                    body=body,
                    summary=node.name.replace("_", " "),
                    query="; ".join(query),
                )
            )
    codes = _error_codes()
    lines = [
        "# HTTP API reference",
        "",
        "<!-- Generated by scripts/generate_reference.py from the FastAPI routes. Do not edit. -->",
        "",
        "Base path `/v1`. Every route except `GET /v1/health` requires the header "
        "`Authorization: Bearer <token>`. See docs/service.md for the security model and the job "
        "states.",
        "",
        "| Method | Path | Auth | Success | Body | Query | Action |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for route in sorted(routes, key=lambda r: (r.path, r.method)):
        auth = "bearer" if route.protected else "none"
        body = f"`{route.body}`" if route.body else ""
        lines.append(
            f"| `{route.method}` | `{route.path}` | {auth} | {route.status} | {body} | "
            f"{route.query} | {route.summary} |"
        )
    lines += ["", "## Error codes", "", "| Code | HTTP status |", "| --- | --- |"]
    lines += [f"| `{code}` | {status} |" for code, status in sorted(codes.items())]
    lines += [
        "",
        'Every error has the body `{"error": {"code", "message", "details"}}`. A job that '
        "ends `failed` carries one of the job error codes instead: `invalid_config`, "
        "`not_found`, `path_not_allowed`, `batch_refused`, `io_error`, `internal` or "
        "`interrupted`.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _error_codes() -> dict[str, int]:
    path = SOURCE / "service" / "errors.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "STATUS_BY_CODE" and isinstance(node.value, ast.Dict):
                return {
                    str(key.value): int(value.value)
                    for key, value in zip(node.value.keys, node.value.values, strict=True)
                    if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
                }
    raise SystemExit("STATUS_BY_CODE not found in errors.py")


def main(argv: list[str]) -> int:
    outputs = {"cli.md": cli_reference(), "http-api.md": http_reference()}
    if "--check" in argv:
        stale = [
            name
            for name, text in outputs.items()
            if not (OUT_DIR / name).is_file()
            or (OUT_DIR / name).read_text(encoding="utf-8") != text
        ]
        if stale:
            print(f"reference out of date: {', '.join(stale)}; run scripts/generate_reference.py")
            return 1
        print("reference is up to date")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (OUT_DIR / name).write_text(text, encoding="utf-8")
        print(f"wrote {OUT_DIR / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
