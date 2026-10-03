# virgil

virgil is a package for modelling optical interferometry data in JAX.

## Installation

```bash
pip install virgil-astro
```

From source:

```bash
pip install .
```

## Using these docs

The tabs at the top group worked examples on simulated and bundled data:
- **Data Handling:** reading OIFITS files into `OIData`, and AMIGO's DISCO data from JWST aperture masking.
- **Sources:** visibility models, extended sources, composing scenes from components, and spotted stars.
- **Binaries:** searching for companions, detection limits, and fitting several datasets together.
- **Imaging:** image reconstruction in five parts, from simulating data to sampling the posterior.

**Background** explains the ideas behind the methods and credits the people and projects virgil builds on. **API Reference** documents every public class and function.

## Development

See the repository contribution guide on GitHub for contribution and testing workflow.

The project is actively being modernized for improved reliability, testing coverage, and model extensibility.

## Name

VIRGIL is the **V**ersatile **I**nterferometric **R**econstruction and **G**radient-based **I**nference **L**ibrary. In Dante's *Divine Comedy*, the poet Virgil is Dante's guide through the Inferno and Purgatory. In Virgil's own *Aeneid*, when Aeneas enters the underworld he draws his sword on the monsters crowding its threshold. His guide, the Cumaean Sibyl, warns him that they are only thin, bodiless lives flitting in a hollow semblance of form (*Aeneid* VI.292–294). Image reconstruction from sparse interferometric data is full of false visions like these: artefacts that look like structure but have no substance in the data. VIRGIL aims to help you tell the difference. The acronym is Jonah Goldfine's.

Until version 0.1.1 the package was named after Voltaire's Dr Pangloss, a nod to Antoine Mérand's [CANDID](https://github.com/amerand/CANDID); it is now distributed on PyPI as `virgil-astro`.

