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
        'torch==1.4.0',
        'bert_serving==0.0.1',
        'Flask==1.1.1',
        'scispacy==0.2.4',
        'Flask_Compress==1.5.0',
        'requests==2.22.0',
        'Flask_JSON==0.3.4',
        'spacy==2.2.3',
        'nltk==3.4.5',
        'numpy==1.16.4',
        'Flask_Cors==3.0.9',
        'transformers==2.5.1',
        'termcolor==1.1.0',
        'flasgger==0.9.4',
        'GPUtil==1.4.0',
        'pymetamap==0.1',
        'pyzmq==19.0.1',
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
