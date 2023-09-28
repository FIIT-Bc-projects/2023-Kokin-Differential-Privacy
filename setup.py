#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Hlib Kokin",
    author_email='xkokin@stuba.sk',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Tuning ML hyperparameters with privacy preservation.",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ml_tuning',
    name='ml_tuning',
    packages=find_packages(include=['ml_tuning', 'ml_tuning.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/xkokin/ml_tuning',
    version='0.0.1',
    zip_safe=False,
)
