from setuptools import find_packages, setup
import setuptools

with open("README.md", "r") as f:
    LONG_DESC = f.read()

setup(name="tsextract", 
      version='0.0.7',
      license='GNU GPL',
      url="https://github.com/cydal/tsExtract/tree/master/tsextract",
      description="Time series data preprocessing", 
      long_description=LONG_DESC,
      long_description_content_type="text/markdown",
      packages=find_packages(include=["tsextract", "tsextract.*"]),
      author="Sijuade Oguntayo", 
      author_email="cydalsij@outlook.com", 
      python_requires=">=3.6",
      install_requires=[
      "pandas >= 1.0.3",
      "seaborn >= 0.10.1",
      "statsmodels >= 0.10.2",
      "scipy >= 1.4.0",
      "matplotlib >= 3.2.1",
      "numpy >= 1.16.4"], 
      classifiers=[
         "Programming Language :: Python :: 3", 
         "Programming Language :: Python :: 3.6", 
         "Programming Language :: Python :: 3.7", 
         "License :: OSI Approved :: GNU General Public License (GPL)", 
         "Operating System :: OS Independent"
      ], 
      zip_safe=False)