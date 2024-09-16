from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name="feature_extraction",
    ext_modules=cythonize(["feature_extraction/*.pyx"]),
    include_dirs=[np.get_include()],
    install_requires=[
        "numpy",
        "scipy",
        "nibabel",
        "cython",
    ],
)