from setuptools import setup
#from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy


sourcefiles = ['./aAMS/aAMS.pyx']

extensions = [Extension("aAMS",sourcefiles,include_dirs=[numpy.get_include()])]


setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
    name='aAMS',
    version='0.0.1',
    description='Adaptive Multilevel Splitting under asymmetric SMC framework with variance estimation.',
    url='https://github.com/MGIMM/aAMS',
    author='Qiming Du',
    author_email='qiming.du@upmc.fr',
    license='MIT',
    packages=['aAMS'],
    install_requires=[
           'tqdm'
           #'joblib',
           #'seaborn'
       ],
    zip_safe=False
)
