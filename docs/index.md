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

The leading optical interferometry model fitting code is [CANDID](https://github.com/amerand/CANDID). In Voltaire's *Candide*, Dr Pangloss' belief that we live in the best of all possible worlds is a satire of Leibniz' theodicy. In a world with JAX, at least we can optimize our fits.