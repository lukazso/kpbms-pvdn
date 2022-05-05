from setuptools import setup, find_packages
import os

lib_dir = os.path.dirname(os.path.realpath(__file__))
requirements_path = lib_dir + "/requirements.txt"
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(
    name='kpsaliency',
    version='0.0',
    packages=find_packages(),
    url='https://github.com/larsOhne/kpsaliency',
    license='',
    author='',
    author_email='',
    description='',
    install_requires=install_requires
)
