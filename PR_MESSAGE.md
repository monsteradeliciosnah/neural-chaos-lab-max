feat(systems): add robust *_step wrappers + exports; fix tests

## Changes
- Implemented *_step wrappers for chaos system functions
- Added safe handling for None and irregular shapes
- Fixed imports for lorenz and rossler to ensure service.py compatibility

## Testing
- All pytest tests now pass locally

## Version
- Bumped to 0.1.1
