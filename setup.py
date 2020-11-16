from setuptools import find_packages, setup
import setuptools

with open("README.md", "r") as f:
    LONG_DESC = f.read()

setup(name="tsextract", 
      version='1.11',
      license='GNU GPL',
      url="https://github.com/cydal/tsExtract/tree/master/tsextract",
      description="Time series preprocessing as supervised learning", 
      long_description=LONG_DESC,
      long_description_content_type="text/markdown",
      #packages=setuptools.find_packages(where='tsextract/*'),
      #packages=["domain", "feature_extraction", "plots"], 
      #package_dir={"": 'tsextract'}, 
      packages=find_packages(include=["tsextract", "tsextract.*"]),
      author="Sijuade Oguntayo", 
      author_email="cydalsij@outlook.com", 
      python_requires=">=3.6",
      zip_safe=False)