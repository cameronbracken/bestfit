# bestfitpy

Python bindings to the [bestfit](https://github.com/cameronbracken/bestfit) C++ core
(a port of the USACE-RMC Numerics / RMC.BestFit libraries) for Bayesian
flood-frequency and extreme-value analysis.

Early development — currently exposes the Generalized Extreme Value distribution:

```python
from bestfitpy import dgev, qgev, gev_fit

qgev(0.99, location=10849, scale=5745.6, shape=0.005)   # ~36977
gev_fit(data, method="mle")                              # {'location':..., 'scale':..., 'shape':...}
```
