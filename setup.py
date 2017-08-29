#!/usr/bin/env python
#   encoding: utf8
#   setup.py
"""DeepWalk python wrapper over native implementation.
"""

from setuptools import find_packages, setup

DOCLINES = (__doc__ or '').split('\n')

CLASSIFIERS = """\
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: C++
Programming Language :: Python
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

PLATFORMS = [
    'Windows',
    'Linux',
    'Solaris',
    'Mac OS-X',
    'Unix',
]

MAJOR = 0
MINOR = 0
PATCH = 0

VERSION = '{0:d}.{1:d}.{2:d}'.format(MAJOR, MINOR, PATCH)


setup(name='deepwalk',
      version=VERSION,
      description = DOCLINES[0],
      long_description = '\n'.join(DOCLINES[2:]),
      url='https://github.com/daskol/deepwalk-c',
      license='MIT',
      platforms=PLATFORMS,
      classifiers=[line for line in CLASSIFIERS.split('\n') if line],
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'convert-bcsr = deepwalk.cli:main',
            ],
      })
