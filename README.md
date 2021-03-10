# quat -- quality analysis tools

collection of python methods and tools as a libary for video/image quality analysis.


## Requirements and Setup

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
poetry install

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
quat = {git="https://github.com/Telecommunication-Telemedia-Assessment/quat.git", branch="master"}
```

## Tools
There are some tools included in `quat`, please checkout the documentation or the command line help:
```bash
poetry run siti --help
poetry run do_parallel --help
poetry run do_parallel_by_file --help
poetry run extract_cuts --help
poetry run psnr --help
poetry run brisque_niqe --help
```


## Note
`quat` is currently not tested under windows, some system specific calls are not working under windows.


## Acknowledgments

If you use this software in your research, please include a link to the repository and reference the following papers.

```
@inproceedings{goering2019qomex,
  author={Steve {G{\"o}ring} and Rakesh Rao {Ramachandra Rao} and Alexander Raake},
  title="nofu - A Lightweight {No-Reference} Pixel Based Video Quality Model for
  Gaming Content",
  BOOKTITLE="2019 Eleventh International Conference on Quality of Multimedia Experience
  (QoMEX) (QoMEX 2019)",
  address="Berlin, Germany",
  days=4,
  month=jun,
  year=2019,
  doi={10.1109/QoMEX.2019.8743262},
  ISSN={2472-7814},
  url={https://ieeexplore.ieee.org/document/8743262},
}
@article{goering2021pixel,
  title={Modular Framework and Instances of Pixel-based Video Quality Models for UHD-1/4K},
  author={Steve G\"oring and Rakesh {Rao Ramachandra Rao} and Bernhard Feiten and Alexander Raake},
  journal={IEEE Access},
  volume={9},
  pages={31842-31864},
  year={2021},
  publisher={IEEE},
  doi={10.1109/ACCESS.2021.3059932},
  url={https://ieeexplore.ieee.org/document/9355144}
}
```

## License
GNU General Public License v3. See [LICENSE.md](./LICENSE.md) file in this repository.
