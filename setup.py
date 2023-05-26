from setuptools import setup, find_namespace_packages

setup(
    name="entity_extractor",
    version="0.1.0",
    author="Ian Bakst",
    author_email="ian.bakst@gmail.com",
    description="The Entity Extractor App",
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(),
    python_requires=">=3.6",
)