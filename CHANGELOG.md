# Changelog

## 0.1.0-rc.1

- Linux installer with an exact engine pin and existing-model SHA256 verification.
- Host checks for single-device Strix Halo gfx1151, 128 GB memory and ROCm access.
- Existing SDK compile/link probe, including recorded rocWMMA 2.2.1 headers.
- `ds4-halo serve` and `run` enable the Halo prefill paths, with explicit context/chunk controls.
- Managed installations retain source and executable hashes; modified builds are refused.
- Updates activate atomically and preserve the previous installation for rollback.
- `doctor`, `status --verify`, command previews and build-only WSL support.

The performance and bitwise results remain the documented historical checkpoint evidence. This packaging revision changes no inference code and supplies no new GPU performance measurement.
