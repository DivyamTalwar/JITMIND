# Assets Directory

This folder holds static media used across documentation, launch material, and architecture communication.

## Purpose

`assets/` is the single source of truth for:

- brand assets
- architecture figures
- README visual systems (root + package + folder-level)
- prompt packs used to generate image sets consistently

Keeping these assets centralized prevents drift in visual language and keeps docs maintainable.

## Structure

```text
assets/
|-- logo.png
`-- readme/
    |-- root_readme_images/               # root README generated image set
    |-- final_svgs/                       # root README animated SVG set
    |-- jitmind_package_images/           # package README generated image set
    |-- jitmind_package_svgs/             # package README animated SVG set
    |-- folder_svgs/                      # lightweight SVGs for tests/eval/examples docs
    |-- theme_samples/                    # early dark-theme visual experiments
    `-- theme_samples_light/              # approved light-theme sample studies
```

## Asset Usage Map

### Root README

- Uses `assets/readme/root_readme_images/*`
- Uses `assets/readme/final_svgs/*`

### Package README (`jitmind/README.md`)

- Uses `assets/readme/jitmind_package_images/*`
- Uses `assets/readme/jitmind_package_svgs/*`

### Folder READMEs

- `tests/README.md` uses `assets/readme/folder_svgs/tests_quality_flow.svg`
- `eval/README.md` uses `assets/readme/folder_svgs/eval_runner_flow.svg`
- `examples/quickstart/README.md` uses `assets/readme/folder_svgs/examples_quickstart_flow.svg`

## Naming Conventions

### SVGs

- Use numeric prefixes for ordered narratives when sequence matters:
  - `01_*`, `02_*`, ...
- Use lowercase snake_case for file names.

### Generated images

- Prefer stable semantic names over timestamp names for README embedding:
  - `01_package_architecture_hero.png`
  - `02_memory_write_lifecycle.png`
  - etc.
- Keep raw timestamp exports if needed, but embed stable aliases in docs.

## Visual System Rules

All current README visuals follow:

- light background
- two primary accents (teal + amber)
- black typography for clarity
- text-heavy technical composition (not decorative poster style)
- smooth and subtle motion for SVG animated flows

This keeps visual coherence across root and package docs.

## Editing Checklist

Before committing asset changes:

1. Verify all referenced paths in README files still resolve.
2. Validate SVG syntax with `xmllint --noout <file.svg>`.
3. Ensure text labels do not overlap in diagrams.
4. Keep consistent terminology with code (`MemoryAgent`, `ResearchAgent`, `RRF`, etc.).
5. Avoid adding large duplicate files unless required for compatibility.

## Notes

- `assets/.DS_Store` and `assets/readme/.DS_Store` are OS-generated and should not be relied on.
- If you add new asset groups, document them here immediately.
