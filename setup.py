from setuptools import setup, find_packages

setup(name='sec', version='1.0', author='andreeanechita',
      author_email='an2u18@soton.ac.uk', packages=find_packages(),
      install_requires=['numpy~=1.19.5', 'scipy~=1.6.1', 'pandas~=1.2.2',
                        'sklearn~=0.0', 'scikit-learn~=0.24.1',
                        'tensorflow~=2.4.1', 'keras-tuner~=1.0.1'],
      description='SmartEnergySecurity for smart energy scheduling and '
                  'detection of manipulated pricing', long_description=open(
        'README.md').read())
