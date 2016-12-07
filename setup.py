from setuptools import setup, find_packages
import os, sys

def readme():
    with open('README.md') as f:
        return f.read()

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__GPROT_SETUP__ = True
import gprot
version = gprot.__version__

# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Push a new tag to GitHub.
if "tag" in sys.argv:
    os.system("git tag -a {0} -m 'version {0}'".format(version))
    os.system("git push --tags")
    sys.exit()

setup(name = "gprot",
    version = version,
    description = "Probabilistic inference of stellar rotation periods.",
    long_description = readme(),
    author = "Ruth Angus, Timothy D. Morton",
    author_email = "ruthangus@gmail.com, tim.morton@gmail.com",
    url = "https://github.com/ruthangus/GProtation",
    packages = ['gprot'],
    package_data = {'gprot':['data/*']},
    scripts = ['scripts/gprot-fit', 'scripts/gprot-acf'],
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    install_requires=[],
    zip_safe=False
)
