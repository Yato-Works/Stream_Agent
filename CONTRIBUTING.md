# Contributing

Thank you for considering a contribution to Stream Agent.

## Development setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -e ".[dev,server]"
python -m pytest -q -m "not network"
```

The network-marked tests contact public model repositories and are not part of
the normal pull request suite. Do not commit model weights, cache contents,
credentials, signed URLs, or personal benchmark data.

## Changes and tests

- Keep storage and network limits explicit; do not silently fall back to full
  artifact downloads.
- Add focused offline tests for correctness and failure handling.
- Update README and evidence documents when supported capabilities or claims
  change. Distinguish measured behavior from planned work.
- Keep model-specific logic in an execution backend or layout adapter, and keep
  generic cache decisions independent of model architecture.
- Run `python -m pytest -q -m "not network"` before submitting.

## Pull requests

Describe the user-visible behavior, relevant tests, and any limitations. For
performance claims, include the model revision, hardware, cache budget, cold or
warm state, and raw measurements. Never include model weights in a patch.

Contributions are provided under the Apache License 2.0, as described in
`LICENSE`.
