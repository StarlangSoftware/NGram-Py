from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='NlpToolkit-NGram',
    version='1.0.16',
    packages=['NGram', 'test'],
    url='https://github.com/StarlangSoftware/NGram-Py',
    license='',
    author='olcaytaner',
    author_email='olcay.yildiz@ozyegin.edu.tr',
    description='NGram library',
    install_requires=['NlpToolkit-DataStructure', 'NlpToolkit-Sampling'],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
