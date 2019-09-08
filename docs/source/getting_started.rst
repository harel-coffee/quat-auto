Getting started
===============

First you need to install the following requirements

- python3 (>=3.5)
- pip3
- ffmpeg (>=3.4.4)

Install it, e.g. using your favorite package-manager, in case of Ubuntu 18.04 use:

.. code-block:: bash

    sudo apt install python3 ffmpeg python3-pip

Clone the `quat` repository, move inside the repository and then you can install `quat` using pip3 via:

.. code-block:: bash

    pip3 install --user -e .

`--user` is for a user wide installation, where `-e` uses this repository as main folder, so changes inside here will be python-wide handled, if this is not required because the repository will be deleted later, just remove `-e` flag.


Hello `quat`
------------
Open the python3 shell and run the following code.

.. code-block:: python3

    from quat.log import *
    lInfo("Hello World")
    lError("an error?")
    jprint({"a": "dictionary"})

If there are no errors happening, than `quat` was successfully installed.


Reading a video
---------------




Included command line tools
---------------------------

- siti.py
- do_parallel.py
- do_parallel_by_file.py


