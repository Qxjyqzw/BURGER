# Shared generation and export pipeline

Start with the repository [README](../README.md) for installation and commands.
[DATA_FORMAT.md](DATA_FORMAT.md) specifies units, frames, observation
timing and end flags.

- `python -m burger generate`: one simulator, one explicit instruction,
  one episode per SVG.
- `python -m burger export`: export existing canonical data.
- `python -m burger import-maniskill`: import supported legacy data.

Use `--help` for current options. See the [verification report](VERIFICATION.md)
for tests and known limitations.
