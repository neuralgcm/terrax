# Xreader: xarray data reader for training AI weather models

Author: Stephan Hoyer

Xreader is a fast data reader for ML training pipelines with data stored in
Xarray. It is built on top of [Grain](https://github.com/google/grain/), and
designed for the needs of training global weather/climate models such as
[NeuralGCM](https://github.com/neuralgcm/neuralgcm) from ARCO datasets stored
in Zarr format such as [ARCO-ERA5](https://github.com/google-research/arco-era5).

## Design goals

1. **Performance**: We need to read data as fast as possible during training, in
   order to ensure that data-hungry accelerators are not idle.
2. **Flexibility**: We need the flexibility to change choices of input
   variables, training time periods and forecasts lengths at training time,
   without needing to regenerate source datasets on disk. Training datasets
   should be stored on disk in "analysis ready" formats, not as as pre-shuffled
   batches of examples.
3. **Introspection**: Inputs and outputs should be fully described Xarray
   objects.

## High level API

- *Inputs*: Nested dictionaries of `xarray.Dataset` objects (technically, a JAX
  [pytree](https://docs.jax.dev/en/latest/pytrees.html) or in the future,
  an `xarray.DataTree`) with a shared dimension name along which to sample,
  typically "time". These datasets only need to share coordinates providing
  locations at which to sample, but otherwise need not be aligned. Typical
  inputs are Xarray datasets pointing to cloud-based Zarr stores that are far
  too large to fit into memory on a single machine.
- *Outputs*: Grain iterators over samples of nested datasets with the exact same
  structure as the inputs, except sliced along the sampled dimension. If
  non-Xarray outputs or batching is desired, this can be achieved with
  additional Grain dataset transformations, e.g., to convert into Coordax.
- *Stencils*: Sampling for each example is specified by `Stencil` objects. A
  nested dictionary of stencils allows for aligned sampling of different windows
  and sampling frequencies for different data parts.
- *Sampling dimensions*: Currently, xreader only supports sampling along one
  dimension (typically "time"). In the future, we hope to add support for
  sampling along multiple dimensions.
