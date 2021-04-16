from setuptools import setup

setup(
    name='pytorch_rrt',
    version='0.1.0',
    packages=['pytorch_rrt'],
    url='https://github.com/UM-ARM-Lab/pytorch_rrt.git',
    license='MIT',
    author='zhsh',
    author_email='zhsh@umich.edu',
    description='Kinodynamic Rapidly-exploring Random Tree (RRT) implemented in pytorch',
    install_requires=[
        'torch',
        'numpy',
        'treelib'
    ],
    tests_require=[
        'gym'
    ]
)
