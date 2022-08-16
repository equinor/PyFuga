from distutils.core import setup

setup(
    name='PyFuga',
    version='0.1dev',
    packages=['fuga', ],
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'xarray',
        'tqdm',
        'h5netcdf',
    ]
)
