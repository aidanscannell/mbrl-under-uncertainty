import pathlib

import setuptools

_here = pathlib.Path(__file__).resolve().parent

name = "mbrllib"
author = "Aidan Scannell"
author_email = "scannell.aidan@gmail.com"
description = "Minimal library for model-based reinforcement learning in PyTorch."

with open(_here / "README.md", "r") as f:
    readme = f.read()

url = "https://github.com/aidanscannell/" + name

license = "Apache-2.0"

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
]
keywords = [
    "model-based-reinforcement-learning",
    "bayesian-deep-learning",
    "deep-learning",
    "machine-learning",
    "bayesian-inference",
    "planning",
]

python_requires = "~=3.7"

install_requires = [
    "torch",
    "functorch",  # needed for vmap
    "numpy",
    "matplotlib",
    "gymnasium",
    "imageio",
    "mujoco",
    "torchtyping",
    "pytorch_lightning",
    # "laplace-torch",
    "gpytorch",
    # "torchrl",
    # f"dmc2gym@ file://{_here}/src/third_party/dmc2gym",
    # f"torchrl@ file://{_here}/src/third_party/rl-0.0.4b",
    # f"torchrl@ file://{_here}/src/third_party/rl",
]
extras_require = {
    "dev": ["black", "pyright", "isort", "pyflakes", "pytest"],
    "experiments": [
        "hydra-core",
        "wandb",
        "hydra-submitit-launcher",
        "dm_control",
        "opencv-python",
        # opencv-python==4.5.3.56
        # "tikzplotlib",
        # "bsuite",
        # "ipython",
        # "seaborn",
    ],
}

setuptools.setup(
    name=name,
    version="0.1.0",
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    keywords=keywords,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    packages=setuptools.find_namespace_packages(),
)
