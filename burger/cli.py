"""Public BURGER commands; simulator modules load only when selected."""
import argparse
import importlib
import sys

COMMANDS = {
    "generate": "burger.pipeline.generate",
    "export": "burger.data.exporters.export",
    "check": "burger.data.check",
    "import-maniskill": "burger.data.import_maniskill",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="burger", description="BURGER robotic drawing data generation",
        epilog="Use burger <command> --help for command-specific options.",
    )
    parser.add_argument("command", choices=COMMANDS, help="generation, export, readback or legacy HDF5 import")
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    previous = sys.argv
    try:
        sys.argv = [f"burger {args.command}", *argv[1:]]
        return importlib.import_module(COMMANDS[args.command]).main()
    finally:
        sys.argv = previous
