# Changelog

## [0.1.1] - 2025-08-11
### Added
- Robust *_step wrappers for chaos system functions
- Safe handling of varied test inputs (random shapes, None, scalars)
- Consistent exports to align with service.py imports

### Fixed
- Test failures in test_step_functions_return_shape due to None outputs
- Import errors for lorenz and rossler systems
