# Plugins

MicroFlow can run third-party algorithms for four jobs: preprocessing an image, segmenting a plane,
measuring features per object, and checking quality. Built-in behaviour is unchanged. A
configuration that uses no plugins behaves, and keeps its cached results, exactly as before.

## Trust model

Read this before installing or enabling a plugin.

- **Installing a package is the trust decision.** A plugin is ordinary Python code. Installing its
  package gives it the same privileges as any dependency: it runs in the MicroFlow process and can
  read and write files. MicroFlow does not sandbox plugins.
- **Discovery runs no plugin code.** `microflow plugins list` and the entry-point lookup read
  package metadata only. Nothing is imported.
- **A plugin runs only when a configuration names it.** Naming a plugin in a configuration or a
  steps file is what enables it. Plugins are referenced from `segmentation.plugin`, from a
  `kind: plugin` preprocessing step, and from the `features` and `quality_checks` lists.
- **Datasets cannot name plugins.** An image record accepts only its name, path, axes, channels
  and calibration. Unknown keys are rejected, so a dataset entry cannot enable code. Image
  files are read as data and nothing in them is interpreted as a plugin name.
- **Plugins never see the caller's data.** MicroFlow passes each plugin a copy of the image, plane,
  or labels. A plugin that changes its input changes only its own copy. Tests check that this
  cannot change the built-in measurements.

## Interfaces

Each plugin is a class with these attributes:

| Attribute | Meaning |
| --- | --- |
| `name` | Entry-point name. Lower-case letters, digits, `-` and `_`, starting with a letter. |
| `version` | The plugin's own version, text. It is recorded in provenance and in the analysis identity. |
| `api_version` | Must equal `PLUGIN_API_VERSION` (currently `1`). |
| `kind` | `preprocessor`, `segmenter`, `feature` or `qc`. Must match the group it is registered under. |
| `options_model` | A pydantic model. It must set `extra="forbid"`, so unknown options are rejected. |

The methods, by kind (see `microflow.plugins.api` for the full signatures):

- **Preprocessor** `process(data, calibration, options) -> array`. `data` is a float64 copy of the
  whole image. Return an array of the same shape. Invalid pixels are NaN. Infinite values are
  refused. A preprocessor that normalises intensities changes what intensity measurements mean,
  so a run that uses one records a warning.
- **Segmenter** `segment(plane, calibration, options) -> integer labels`. `plane` is a float64
  copy of one Y-X plane. Return labels of the same shape: 0 is background, objects are positive,
  and no label may exceed the number of pixels. MicroFlow then applies the minimum-area and
  border rules and renumbers the labels in raster order.
- **Feature** `features() -> (FeatureColumn(name, unit, description), ...)` and
  `measure(labels, plane, calibration, options) -> {label: {feature: value}}`. The result must
  cover every object in the plane, and only objects. Values are finite numbers, or NaN. Each
  feature plugin writes its own `features_<name>.csv` and a data dictionary for it.
- **Quality check** `check(labels, plane, calibration, options) -> [message, ...]`. Each message is
  recorded as a warning on the image, prefixed with the plugin and plane. Findings do not fail the
  image.

`calibration` is the image's `Calibration`, or `None` when it has none. Plugins receive it so they
can convert lengths; they must not assume pixels are micrometres.

## Output contracts

Each output is checked before MicroFlow uses it. A violation fails the image the plugin ran on,
and the error names the plugin and the broken rule.

| Kind | Required of the output |
| --- | --- |
| Preprocessor | Same shape as the input; numeric; no infinite values. |
| Segmenter | Same shape as the plane; integer; no negative labels; no label above the pixel count. |
| Feature | Exactly the declared features, for exactly the objects in the plane; keys are integers; values are finite numbers or NaN. Declared features have unique names and non-empty units and descriptions. |
| Quality check | A sequence of non-empty text messages. |

Exceptions raised by a plugin, including `SystemExit`, are reported as `PluginError`. Keyboard
interrupts are not caught, so Ctrl-C still cancels a run.

## Failure behaviour

- **Plugin not installed, or more than one package provides its name:** the command or batch
  refuses before it writes anything. The message lists what is installed.
- **Invalid options, a declaration that breaks the interface, or an import or constructor error:**
  the same as above. The message names the plugin and the field or rule.
- **A contract violation or exception during a run:** only the image the plugin ran on fails. Its
  partial outputs are removed, its result is recorded as failed with the plugin's error, and other
  images are unaffected. The run finishes as `completed_with_failures`.
- **Quality findings:** recorded as warnings. The image still completes.

Nothing a plugin returns is written to disk unless it passes its contract. A failed image publishes
no partial output.

## Configuration

```yaml
preprocessing:
  - kind: plugin
    plugin: example-offset
    options: {offset: 10.0}
segmentation:
  method: plugin
  plugin: {name: example-quantile, options: {quantile: 0.9}}
  min_object_area: {value: 4, unit: um}
features:
  - name: example-object-range
quality_checks:
  - name: example-small-fraction
    options: {min_pixels: 4, max_fraction: 0.5}
```

With `method: plugin`, the built-in segmentation options (`threshold`, `adaptive_*`, `opening_radius`,
`closing_radius`, `watershed*`) must be absent. Those options do not apply to plugin segmentation,
and MicroFlow refuses a configuration that sets them.

## Identity, provenance and caching

- Every named plugin's kind, name, version and validated options join the analysis identity.
  Changing an option or a version changes the identity, so an existing output directory is refused
  until overwritten. The same rule applies to any other analysis setting.
- Configurations without plugins keep the identity they had before plugins existed, so their
  results stay reusable.
- Each run's provenance lists the version of each plugin under `algorithm_versions` as
  `plugin:<kind>:<name>`.
- **Limit:** MicroFlow cannot detect a change in a plugin's code that is not reflected in its
  `version`. Bump `version` whenever the output for the same input and options changes.

## Commands

- `microflow plugins list` lists installed plugins by kind, name, distribution and target. It runs
  no plugin code.
- `microflow plugins check CONFIG` loads every plugin the configuration names and validates its
  options. It runs no images.
- `microflow batch`, `microflow segment` and `microflow preprocess` load the plugins they use and
  report plugin errors with exit status 1.
- `microflow compare-configs` and `microflow cache-status` load the plugins a configuration names,
  because the analysis identity includes their versions. They change nothing.

## Writing a plugin

Register the class under the group for its kind in your package's `pyproject.toml`:

```toml
[project.entry-points."microflow.segmenters"]
my-segmenter = "my_package.segmentation:MySegmenter"
```

The groups are `microflow.preprocessors`, `microflow.segmenters`, `microflow.features` and
`microflow.qc_checks`. Depend on `microflow` for the interfaces, and check your plugin against the
contracts with the tests in `tests/test_plugins.py` as a model.

## Example plugin

`examples/plugins/microflow-example-plugins` is one distribution with one plugin per kind:

| Kind | Name | Behaviour |
| --- | --- | --- |
| Preprocessor | `example-offset` | Adds a constant offset to every valid pixel. |
| Segmenter | `example-quantile` | Foreground is the set of pixels above an intensity quantile. |
| Feature | `example-object-range` | Intensity range (maximum minus minimum) of each object. |
| Quality check | `example-small-fraction` | Warns when too many objects are smaller than a size. |

Install it into the project environment with uv. Use `--no-deps`, because the package depends on
`microflow` itself, which is already installed:

```sh
uv pip install --no-deps ./examples/plugins/microflow-example-plugins
```

## Verification

- `tests/test_plugins.py` covers discovery, load-time validation, option validation, and every
  output contract, including exception wrapping and keyboard-interrupt passthrough.
- `tests/test_plugin_integration.py` runs plugins through segmentation, preprocessing and batch
  runs. It checks that a failing plugin fails only its image, that no partial output is published,
  that a plugin cannot change built-in measurements, and that plugin settings enter the identity.
- `tests/test_example_plugin.py` checks the example's declaration against its code and runs the
  example through the same registration path a third-party package uses. It also installs the
  example with uv into a temporary directory. That last test is skipped where uv is not available.
- `tests/test_cli_plugins.py` covers `plugins list`, `plugins check`, and plugin errors from
  `batch`.

## Limits

- Plugins run in-process. There are no timeouts and no per-plugin memory limits.
- A plugin receives a copy of its input, so a large image is copied once per plugin call.
- Per-plane plugins are called once per Y-X plane, and their configuration is loaded once per plane.
- The trust model above is about which code runs. It is not a security boundary against hostile
  code: an installed plugin can do anything the MicroFlow process can.
