# quat -- quality analysis tools

collection of python methods and tools as a libary for video/image quality analysis.


## requirements and setup

first you need:

* python3 (>=3.6)
* pip3
* ffmpeg
* poetry (`pip3 install --user poetry`)

install it, e.g. via your package-manager, in case of ubuntu use:

```bash
sudo apt install python3 ffmpeg python3-pip
pip3 install --user poetry
```

then you can install `quat` using pip via:

```bash
poetry build
pip3 install dist/*.whl
```

For development you can also just stay in the poetry environment and run specifc parts, e.g. with
```bash
poetry run siti --help
```

## quat as dependency
You can also use `quat` as a dependency in a poetry project, e.g. adding the following to your projects `pyproject.toml`:

```ini
quat = {git="git@github.com:stg7/quat.git", branch="master"}
```

## tools
There are some tools included in `quat`, please checkout the documentation or the command line help:
```bash
poetry run siti --help
poetry run do_parallel --help
poetry run do_parallel_by_file --help
poetry run extract_cuts --help
poetry run psnr --help
```


## note
`quat` is currently not tested under windows, some system specific calls are not working under windows.

## local ffmpeg
to use a local installed ffmpeg/ffprobe version, please ..

