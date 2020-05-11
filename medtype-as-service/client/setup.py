from os import path

from setuptools import setup, find_packages

# setup metainfo
libinfo_py = path.join('medtype_serving', 'client', '__init__.py')
libinfo_content = open(libinfo_py, 'r').readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
exec(version_line)  # produce __version__

with open('requirements.txt') as f:
    require_packages = [line[:-1] if line[-1] == '\n' else line for line in f]

setup(
    name='medtype_serving_client',
    version=__version__,  # noqa
    description='Performs Medical Entity Linking on text using MedType (Client)',
    url='https://github.com/svjan5/medtype',
    long_description=open('Readme.md', 'r').read(),
    long_description_content_type='text/markdown',
    author='Shikhar Vashishth',
    author_email='shikharvashishth@gmail.com',
    license='Apache 2.0',
    packages=find_packages(),
    zip_safe=False,
    install_requires=require_packages,
    classifiers=(
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apaceh 2.0 License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
    keywords='medtype nlp bionlp pytorch machine learning medical entity linking',
)
