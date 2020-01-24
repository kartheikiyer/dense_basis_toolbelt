# dense_basis_toolbelt

This is a place for experimental addons to the `dense_basis` SED fitting code to live. Documentation and usage can be found at the [dense_basis](https://dense-basis.readthedocs.io) docs. The main reason for making this a separate package is the additional dependences this needs (including, but not limited to: pytorch, ranger, emcee, sklearn and joblib)

This includes
- Sampling methods (e.g., MCMC, Nested Sampling and more)
- Optimization techinques (e.g., efficient parallelization, a NN+PCA backend (similar to [Speculator; Alsing et al. 2019](https://arxiv.org/abs/1911.11778)) or NDinterpolation coupled to the inference tools)
- More visualization/interpretation tools
