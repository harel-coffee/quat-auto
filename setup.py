import setuptools
from distutils.core import setup

setup(
    name='quat',
    version='0.1',
    description='quality analysis tools',
    author='Steve Göring',
    author_email='stg7@gmx.de',
    packages=['quat'],
    install_requires=[
        "scikit-image",
        "scikit-video",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn"
    ],
)