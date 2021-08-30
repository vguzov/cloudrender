from setuptools import setup
version = '1.0.0'

with open("README.md", "r") as fi:
    long_description = fi.read()

keywords = ["rendering", "pointcloud", "opengl", "mesh"]

classifiers = [
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3'
    ]

requirements = [
        "numpy",
        "scipy",
        "pyglm>=2.2.0",
        "trimesh",
        "tqdm",
        "smplpytorch",
        "PyOpenGL==3.1.5",
        "videoio>=0.2.3"
]

setup(
    name="cloudrender",
    packages=["cloudrender"],
    version=version,
    description="An OpenGL framework for pointcloud and mesh rendering",
    author="Vladimir Guzov",
    author_email="vguzov@mpi-inf.mpg.de",
    url="https://github.com/vguzov/cloudrender",
    keywords=keywords,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    classifiers=classifiers
)