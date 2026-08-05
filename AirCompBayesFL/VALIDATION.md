# Validation

Version 1.2.0 validation performed in the packaging environment:

- All Python modules compile successfully.
- Eight unit tests pass, including AirComp, aggregation, model size, runtime
  backend selection, and wireless distance normalization tests.
- A numerical AirComp check confirms that normalized-distance channels avoid
  the near-total update suppression produced by the legacy raw-metre channel.

An end-to-end RTX 3060 run cannot be executed in this packaging environment.
The user's prior v1.1 run established that CUDA 12.1, local client training,
and ideal/no-wireless aggregation execute successfully on their machine.
