"""
cli.py — Command-line interface for AxonForge.

Usage:
    axonforge [graph_name]         Launch the GUI for the current project.
    axonforge init [--name NAME]   Create a new project in the current directory.
    axonforge docs TOPIC           Print built-in authoring help for external users and agents.
    axonforge validate-node PATH   Validate a node definition file.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from axonforge.project import DIR_NODES, DIR_GRAPHS, DIR_DATA

PROJECT_FILE = "axonforge.json"


def cmd_init(args: argparse.Namespace) -> None:
    cwd = Path.cwd()
    project_file = cwd / PROJECT_FILE

    if project_file.exists():
        print(f"Error: {PROJECT_FILE} already exists in this directory.")
        sys.exit(1)

    name = args.name or cwd.name

    for folder in (DIR_NODES, DIR_GRAPHS, DIR_DATA):
        (cwd / folder).mkdir(exist_ok=True)

    project_meta = {
        "version": 1,
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    project_file.write_text(json.dumps(project_meta, indent=2) + "\n")

    print(f"Initialized AxonForge project '{name}' in {cwd}")
    print(f"  {DIR_NODES}/  — place your custom node files here")
    print(f"  {DIR_GRAPHS}/ — node graphs are saved here")
    print(f"  {DIR_DATA}/   — numpy array and dataset storage")


def cmd_run(args: argparse.Namespace) -> None:
    cwd = Path.cwd()
    project_file = cwd / PROJECT_FILE

    if not project_file.exists():
        print(f"Error: {PROJECT_FILE} not found. Run 'axonforge init' first.")
        sys.exit(1)

    graph_name = args.graph_name

    # Validate graph exists if specified
    if graph_name:
        graph_path = cwd / DIR_GRAPHS / f"{graph_name}.json"
        if not graph_path.exists():
            print(f"Error: Graph '{graph_name}' not found at {graph_path}")
            sys.exit(1)

    from axonforge_qt.main import main
    main(project_dir=cwd, graph_name=graph_name)


def cmd_validate_node(args: argparse.Namespace) -> None:
    from axonforge.validation import print_validation_result, validate_node_target

    result = validate_node_target(
        args.path,
        class_name=args.class_name,
        project_dir=args.project_dir,
        run_init=args.run_init,
    )
    print_validation_result(result, as_json=args.json)
    sys.exit(0 if result.get("ok") else 1)


def cmd_docs(args: argparse.Namespace) -> None:
    from axonforge.docs_cli import list_doc_topics, render_doc_topic

    if args.topic is None:
        topics = ", ".join(list_doc_topics())
        print(f"Available docs topics: {topics}")
        sys.exit(0)

    try:
        output = render_doc_topic(args.topic, as_json=args.json)
    except KeyError:
        topics = ", ".join(list_doc_topics())
        print(f"Error: unknown docs topic '{args.topic}'. Available topics: {topics}")
        sys.exit(1)

    print(output)


SUBCOMMANDS = {"init", "docs", "validate-node", "-h", "--help"}


def _build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axonforge",
        description="AxonForge — node-based visual editor for bio-inspired learning algorithms",
    )
    subparsers = parser.add_subparsers(dest="command")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize a new project in the current directory")
    init_parser.add_argument("--name", type=str, default=None, help="Project name (defaults to directory name)")

    # docs
    docs_parser = subparsers.add_parser(
        "docs",
        help="Print built-in framework documentation topics",
    )
    docs_parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="One of: agent, template, fields",
    )
    docs_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output",
    )

    # validate-node
    validate_parser = subparsers.add_parser(
        "validate-node",
        help="Validate a built-in or project-local node definition file",
    )
    validate_parser.add_argument("path", help="Path to the node .py file")
    validate_parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Validate only a specific Node subclass in the file",
    )
    validate_parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Project root to use for project-local node discovery (defaults to cwd if axonforge.json exists)",
    )
    validate_parser.add_argument(
        "--run-init",
        action="store_true",
        help="Also call init() after structural validation",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output",
    )

    return parser


def main() -> None:
    # If the first argument is a known subcommand (or help flag), use subcommand parsing.
    # Otherwise, treat bare `axonforge [graph_name]` as the run/launch command.
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS:
        parser = _build_subcommand_parser()
        args = parser.parse_args()

        if args.command == "init":
            cmd_init(args)
        elif args.command == "docs":
            cmd_docs(args)
        elif args.command == "validate-node":
            cmd_validate_node(args)
    else:
        parser = argparse.ArgumentParser(
            prog="axonforge",
            description="AxonForge — node-based visual editor for bio-inspired learning algorithms",
        )
        parser.add_argument("graph_name", nargs="?", default=None, help="Graph to open on startup")
        args = parser.parse_args()
        cmd_run(args)


if __name__ == "__main__":
    main()
