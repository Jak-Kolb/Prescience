# scripts

Thin command-line entrypoints.

## Philosophy
Scripts should do:
1) parse CLI args
2) call `prescience.pipeline.*`
3) exit

## Examples
- `count_webcam.py` -> `pipeline.count_stream`
- `count_video.py`  -> `pipeline.count_video`
- `enroll.py`       -> `pipeline.enroll`
