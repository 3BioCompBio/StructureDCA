# StructureDCA Release Notes

## Version 1.1.x

This update preserves the overall behavior of StructureDCA while introducing small implementation and output changes.

- Structure parsing now relies entirely on Biopython.
- Added support for `.cif` (mmCIF) files as 3D structure inputs.
- When `ignore_hydrogen_atoms=True`, hydrogen atoms are also ignored when calculating Relative Solvent Accessibility (RSA). This behavior was missing in version 1.0.x and can produce very small changes in `StructureDCA[RSA]` predictions.

## Version 1.0.x

- First release of the StructureDCA Python package, accompanying the first publication on StructureDCA: [Matsvei Tsishyn, Hugo Talibart, Marianne Rooman, Fabrizio Pucci. Structure-informed direct coupling analysis improves protein mutational landscape predictions. BioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.27.714804v1)
