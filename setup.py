'''
A collection of Data Science helper functions
'''

import setuptools

REQUIRED = [
    'numpy'
    'pandas'
]

with open("README.md', 'r') as file:
    LONG_DESCRIPTION = file.read()

setuptools.setup(
    name='lambdata-SWDS17',
    version='0.0.1',
    author='StevenWestmoreland',
    description='A collection of Data Science helper functions',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/StevenWestmoreland/LambdataDS17',
    packages+setuptools.find_Packages(),
    python_requires='>==3.7',
    install_requires=REQUIRED,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)