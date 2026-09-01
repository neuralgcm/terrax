# Terrax documentation

## Overview

Terrax is a Python library for **building, training, and running** AI-first
Earth System Models (ESMs) and their components. It is built on
[JAX](https://github.com/jax-ml/jax),
[NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html),
[Coordax](https://github.com/neuralgcm/coordax), and
[xarray](https://xarray.pydata.org), enabling efficient hardware acceleration,
powerful automatic differentiation, and seamless integration with the
open-source geospatial ecosystem.

Key concepts of the Terrax API include:

* A data model using `coordax.Field` structures that are convertible to and
  from `xarray.Dataset` for easy inspection and serialization.
* A `Model` API for defining models by subclassing `api.Model` and implementing
  `assimilate`, `advance`, and `observe` methods.
* A typing system to manage simulation state components like prognostics,
  diagnostics, randomness, and dynamic inputs (forcings).
* An immutable and purely functional `InferenceModel` API for running forecasts,
  compatible with JAX transformations and scalable inference with Apache Beam.
* A `VectorizedModel` for efficient batch and ensemble simulations.

```{tip}
To stay up to date on Terrax, check out the
[GitHub repository](https://github.com/neuralgcm/terrax).
```

## Questions?

The best place to ask for help using Terrax is
[on GitHub](https://github.com/neuralgcm/terrax/issues).

## Contents

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation.md
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

data_model_tutorial.ipynb
model_api_tutorial.ipynb
simulation_state_components_tutorial.ipynb
inference_model_api_tutorial.ipynb
model_vectorization_tutorial.ipynb
transforms_tutorial.ipynb
layers_towers_transforms_tutorial.ipynb
```

```{toctree}
:maxdepth: 1
:caption: Demos

learning_l96_demo.ipynb
forced_parameterized_coupled_l96.ipynb
```
