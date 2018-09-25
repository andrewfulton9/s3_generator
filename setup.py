from setuptools import setup, find_packages

setup(
    name='S3Generator',
    version='0.1',
    packages=find_packages(),
    scripts=['s3_generator.py'],

    install_requires=['keras', 'boto3'],

    author="Andrew Fulton",
    auther_email='andrewfulton9@gmail.com',
    description='Module for loading images from s3 into a neural net',
    license='MIT',
    keyworks='S3 generator keras neural net',
    project_urls = {
        'Bug Tracker': '',
        'Source Code': ''
    }
)