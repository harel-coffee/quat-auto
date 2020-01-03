# quat -- quality analysis tools

collection of python methods and tools as a libary for video/image quality analysis.

## requirements and setup

first you need:

* python3 (>=3.6)
* pip3
* ffmpeg

install it, e.g. via your package-manager, in case of ubuntu use:

```bash
sudo apt install python3 ffmpeg python3-pip
```

then you can install `quat` using pip via:

```bash
pip3 install --user -e .
```
`--user` is for a user wide installation, where `-e` uses this repository as main folder, so changes inside here will be python-wide handled.

## note
* `quat` is currently not tested under windows, some system specific calls are not working under windows.

## local ffmpeg
to use a local installed ffmpeg/ffprobe version, please ..

