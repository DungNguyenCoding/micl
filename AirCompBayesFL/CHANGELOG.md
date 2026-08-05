# Changelog

## 1.2.0

- Corrected the wireless path-loss unit mismatch by making the distance
  reference explicit (`wireless.path_loss_reference_m`).
- Defaulted the disclosed-paper configurations to 1000 m, equivalent to using
  kilometres in the dimensionless `r^{-alpha}` channel law.
- Added AirComp diagnostics for retained update magnitude and received/ideal
  norm ratio.
- Made clipping statistics ignore numerically insignificant attenuation.
- Preserved native-Windows sequential CUDA execution and Linux/WSL2 Ray mode.

## 1.1.0

- Added stable native-Windows local CUDA backend.
- Abort runs on failed clients instead of logging invalid random-model results.
