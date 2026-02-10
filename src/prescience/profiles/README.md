# profiles

Defines and persists SKU identity information (“product profiles”).

## Responsibilities
- `schema.py`: versioned profile structure
- `io.py`    : load/save profiles to disk

## Notes
Keep formats versioned so older profiles remain readable as the project evolves.
