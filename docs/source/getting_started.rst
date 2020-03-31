Getting started
===============

First you need to install the following requirements

- python3 (>=3.6.1)
- pip3
- ffmpeg (>=3.4.4)
- poetry

Install it, e.g. using your favorite package-manager, in case of Ubuntu >=18.04 use:

.. code-block:: bash

    sudo apt install python3 ffmpeg python3-pip
    pip3 install --user poetry

Clone the `quat` repository, move inside the repository and then you can install `quat` using pip3 via:

.. code-block:: bash

    poetry build
    pip3 install dist/*.whl

Beside this you can also just use `quat` with the poetry tool.


Hello `quat`
------------
Open the python3 shell and run the following code.

.. code-block:: python3

    from quat.log import *
    lInfo("Hello quat")
    lError("an error?")
    jprint({"a": "dictionary"})

If there are no errors happening, than `quat` was successfully installed.


Reading a video
---------------

.. code-block:: python3

    from quat.log import *
    from quat.video import iterate_by_frame
    from quat.visual.base_features import SiFeatures

    si = SiFeatures()
    video = "path/to/video"
    for frame in iterate_by_frame(video):
        lInfo(frame)
        v = si.calc(frame)
        jprint({"si": v})



Included command line tools
---------------------------

All of the following command line tools are globally available after installation of `quat`, see also :ref:`module tools`

- siti : calculation of siti feature values for a given video
- do_parallel : perform parallel tasks
- do_parallel_by_file : perform parallel tasks
- extract_cuts : extract scene cuts of a given video
- psnr : calculates psnr for a distorted and source video.

All command line tools can be started with `poetry run`.


