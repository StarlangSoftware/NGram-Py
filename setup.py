from setuptools import setup

setup(
    name='NlpToolkit-NGram',
    version='1.0.13',
    packages=['NGram', 'test'],
    url='https://github.com/olcaytaner/NGram-Py',
    license='',
    author='olcaytaner',
    author_email='olcaytaner@isikun.edu.tr',
    description='NGram library',
    install_requires=['NlpToolkit-DataStructure', 'NlpToolkit-Sampling']
)
