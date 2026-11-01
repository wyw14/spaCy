Bug fix for model downloading in environments without pip on PATH

## Features and enhancements

- Add `Language.timing` to process a single text or `Doc` once and return the processed `Doc` together with an ordered report of monotonic wall-clock timings for tokenization and each actually executed pipeline component, supporting the same `disable` and `component_cfg` arguments as a normal call

## Fixes

- Fix `spacy download` failing in environments where `pip` is not on PATH but is available as a Python module (e.g., some virtual environments and containers)
