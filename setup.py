from setuptools import setup, find_packages
import sys

if sys.version_info<(3,):
    sys.exit("Sorry, Python 3 is required for Caver")

with open("README.md", "r") as f:
    readme = f.read()

with open("requirements.txt", "r") as f:
    reqs = [l for l in f.read().splitlines() if l]
    print(reqs)

setup(
    name="brief",
    version="0.0.1",
    description="Text Summarizer",
    long_description=readme,
    author='Guokr Inc.',
    author_email='jinyang.zhou@guokr.com',
    url="https://github.com/guokr/Anutshell",
    packages=find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ),
    entry_points={
        'console_scripts': [
            # 'trickster_train=trickster::train',
        ]
    },
    install_requires=reqs
)
