from setuptools import setup

setup(
    name='NlpToolkit-NGram',
    version='1.0.14',
    packages=['NGram', 'test'],
    url='https://github.com/StarlangSoftware/NGram-Py',
    license='',
    author='olcaytaner',
    author_email='olcay.yildiz@ozyegin.edu.tr',
    description='NGram library',
    install_requires=['NlpToolkit-DataStructure', 'NlpToolkit-Sampling']
)
