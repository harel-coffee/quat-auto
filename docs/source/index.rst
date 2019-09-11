quat -- video/image quality analysis
====================================
`quat` is a collection of tools, python3 code, and wrappers, to simplify daily quality analysis tasks for video or images.
`quat` is the outcome of several video/image quality models that were developed by Steve Göring.

It consists of several main modules, with specific tasks,
e.g. the `ml` module handles typical machine learning parts, whereas most algorithms and approaches are based on scikit-learn.


Why you should or should not use `quat`?
----------------------------------------
`quat` is a collection of useful parts, it is not a full end-solution or framework that will handle automatically all image/video quality related questions.
You can use `quat` to develop own models, to extract image/video features, and more.


Reference
---------
If you use `quat` in any research related project, please cite the following paper:

.. code-block:: bibtex

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


`nofu` is a video quality model using features and general video processing methods that are included in `quad`.

Projects using `quat`
---------------------

- nofu
- hyfu
- fume
- hyfr
- uhdhd prediction

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   getting_started.rst
   examples.rst
   api.rst

