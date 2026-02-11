# events

Shared edge/cloud event contracts and transport helpers.

## Event Types

- `COUNT`
- `HEARTBEAT`
- `ALERT`

Each event carries `seq`, `timestamp`, `frame_ts`, `line_id`, `device_id`, and optional `run_id`.
