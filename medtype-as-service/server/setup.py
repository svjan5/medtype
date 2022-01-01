from os import path

from setuptools import setup, find_packages

# setup metainfo
libinfo_py = path.join('medtype_serving', 'server', '__init__.py')
libinfo_content = open(libinfo_py, 'r').readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
exec(version_line)  # produce __version__

setup(
    name='medtype_serving_server',
    version=__version__,
    description='Performs Medical Entity Linking on text using MedType (Server)',
    url='https://github.com/svjan5/medtype',
    long_description=open('Readme.md', 'r').read(),
    long_description_content_type='text/markdown',
    author='Shikhar Vashishth',
    author_email='shikharvashishth@gmail.com',
    license='Apache 2.0',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'six',
        'torch',
        'bert_serving',
        'Flask',
        'scispacy',
        'Flask_Compress',
        'requests',
        'Flask_JSON',
        'spacy',
        'nltk',
        'numpy',
        'Flask_Cors',
        'transformers',
        'termcolor',
        'flasgger',
        'GPUtil',
        'pymetamap',
        'pyzmq',
    ],
    extras_require={
        'http': ['flask', 'flask-compress', 'flask-cors', 'flask-json', 'medtype-serving-client']
    },
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apace 2.0 License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
    entry_points={
        'console_scripts': ['medtype-serving-start=medtype_serving.server.cli:main',
                            'medtype-serving-benchmark=medtype_serving.server.cli:benchmark',
                            'medtype-serving-terminate=medtype_serving.server.cli:terminate'],
    },
    keywords='medtype nlp bionlp pytorch machine learning medical entity linking',
)
