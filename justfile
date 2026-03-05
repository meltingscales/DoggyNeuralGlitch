default:
    @just --list

sync:
    uv sync

test:
    uv run test.py

# Generate glitch art frames and combine with audio into output/output.mp4.
# fps should match 1/chunk-duration (default chunk-duration=0.5 -> fps=2).
# Modes: neural, chaos, mixed, bitcrush, rgb_split, datamosh
run input output="./output" mode="mixed" fps="2":
    uv run main.py --input {{input}} --output {{output}} --mode {{mode}}
    ffmpeg -y -framerate {{fps}} -i {{output}}/frame_%04d.png -i {{input}} \
        -c:v libx264 -c:a aac -pix_fmt yuv420p -shortest \
        {{output}}/output.mp4

# Same as `run` but limits frame count. Useful for previewing long audio files.
run-frames input frames output="./output" mode="mixed" fps="2":
    uv run main.py --input {{input}} --output {{output}} --mode {{mode}} --max-frames {{frames}}
    ffmpeg -y -framerate {{fps}} -i {{output}}/frame_%04d.png -i {{input}} \
        -c:v libx264 -c:a aac -pix_fmt yuv420p -shortest \
        {{output}}/output.mp4

clean:
    rm -rf ./output ./test_output

clean-all: clean
    rm -rf .venv uv.lock
