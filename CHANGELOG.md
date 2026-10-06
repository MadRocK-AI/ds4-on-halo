# Changelog

## 0.1.0-rc.2 — 2026-10-06

- Recorded the separate patched engine candidate's fresh 4K numerical/cache-lifetime PASS and IOMMU-off prepared **454.59 token/s** mean (**+44.12%** versus its fresh reference), passing the unchanged 440 minimum.
- Complete compared on/off payloads remained bitwise identical; candidate mean improved 5.77% over its retained 429.79 on-mode result. The cleaned pinned source passed real installer, complete installed numerical and launcher/API acceptance. Installer scripts are unchanged; only the source pin, version and qualification evidence advance from rc.1.

## 0.1.0-rc.1

- Linux installer with an exact engine pin and existing-model SHA256 verification.
- Host checks for single-device Strix Halo gfx1151, 128 GB memory and ROCm access.
- Existing SDK compile/link probe, including recorded rocWMMA 2.2.1 headers.
- `ds4-halo serve` and `run` enable the Halo prefill paths, with explicit context/chunk controls.
- Managed installations retain source and executable hashes; modified builds are refused.
- Updates activate atomically and preserve the previous installation for rollback.
- `doctor`, `status --verify`, command previews and build-only WSL support.

The performance and bitwise results remain the documented historical checkpoint evidence. This packaging revision changes no inference code and supplies no new GPU performance measurement.

Post-publication validation on 2026-10-05: the public pinned package installed, passed model/source/binary checks and served two live API requests on Halo. The prompts contained 14 and 4,214 tokens with 2K chunks; eight tokens were generated per request before the thinking budget ended. The owned process and exclusive hardware lock were released. [Live record](live-smoke.json).
