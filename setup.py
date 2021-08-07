from setuptools import setup, find_packages

__version__ = '0.0.1'
url = 'https://github.com/mrdrozdov/pytorch_particle'

install_requires = []
setup_requires = []
tests_require = ['pytest', 'pytest-runner', 'pytest-cov', 'mock']

setup(
    name='torch_particle',
    version=__version__,
    description='',
    author='Andrew Drozdov',
    author_email='andrew@mrdrozdov.com',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[],
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require},
    packages=find_packages(),
    include_package_data=True,
)
