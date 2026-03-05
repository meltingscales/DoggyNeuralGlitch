default:
    @just --list

sync:
    uv sync

test:
    uv run test.py

# Generate glitch art frames at 30fps (chunk-duration=0.033s) and combine with
# audio into output/output.mp4. Override fps/chunk-duration together if you change
# one: chunk-duration = 1/fps (e.g. fps=2 -> chunk-duration=0.5).
# Modes: neural, chaos, mixed, bitcrush, rgb_split, datamosh, drakonix
run input output="./output" mode="mixed" fps="30" chunk-duration="0.0333":
    uv run main.py --input {{input}} --output {{output}} --mode {{mode}} --chunk-duration {{chunk-duration}}
    ffmpeg -y -framerate {{fps}} -i {{output}}/frame_%04d.png -i {{input}} \
        -c:v libx264 -c:a aac -pix_fmt yuv420p -shortest \
        {{output}}/output.mp4

# Same as `run` but limits frame count. Useful for previewing long audio files.
run-frames input frames output="./output" mode="mixed" fps="30" chunk-duration="0.0333":
    uv run main.py --input {{input}} --output {{output}} --mode {{mode}} --max-frames {{frames}} --chunk-duration {{chunk-duration}}
    ffmpeg -y -framerate {{fps}} -i {{output}}/frame_%04d.png -i {{input}} \
        -c:v libx264 -c:a aac -pix_fmt yuv420p -shortest \
        {{output}}/output.mp4

# Convenience targets — one per main mode, all at 30fps.
run-mixed input output="./output":
    just run {{input}} {{output}} mixed

run-chaos input output="./output":
    just run {{input}} {{output}} chaos

run-neural input output="./output":
    just run {{input}} {{output}} neural

run-drakonix input output="./output":
    just run {{input}} {{output}} drakonix

clean:
    rm -rf ./output ./test_output

clean-all: clean
    rm -rf .venv uv.lock
