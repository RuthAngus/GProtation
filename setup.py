from setuptools import setup

setup(name='gprotation',
      version='0.1',
      description='Probabilistic rotation period inference',
      url='http://github.com/RuthAngus/gprotation',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['gprotation'],
      install_requires=['numpy', 'matplotlib', 'pandas', 'h5py', 'emcee',
                        'emcee3', 'gatspy', 'george', 'corner'],
      zip_safe=False)
