# Verification record

Date: 2026-10-19 (Asia/Singapore).

- Existing OpenPI protocol, final audit, archive verifier, plotting and continuous-data tests: **24 passed in 10.51s**. Run used `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, and the existing OpenPI environment; no simulation/training/inference.
- All 170 staged/new Python source files parsed successfully before final packaging.
- 45 retained HDF5 sources: 409,670 frames; 50,799,080 numeric values checked in 2,048-row chunks, with no nonfinite/length issue. Video container frame-count metadata matches all 45 sources.
- Independent export alignment: all A/B physical rows match reconstructed raw state/action exactly. Valid anchor boundaries, source splits, sentinel rows and nominal frame/30 timestamps checked; actual sensor/exposure/apply clocks not verified.
- Independent rescoring: 9,780 saved prediction chunks; checked metrics agree exactly with archived records. Training/inference/rollout remain Not independently verified.
- OpenPI patch: dry-run succeeds for 82 changed/new files; source snapshot validation records 223 files, including 6 sanitized YAML templates.
- Non-video figures regenerated using only repository numeric tables. New plots visually inspected; no synthetic measurement or AI-generated task image.
- Word v7: 18 native tables, 9 numbered figures plus the cover image, valid package/XML, original input hash unchanged; set to A4. No Word/LibreOffice renderer available, so pagination remains to be checked in Word after field update.

Exact final staged-file safety results are in SECURITY_CHECKS.json. A regex safety scan is one check, not a guarantee against every possible secret.
