from setuptools import setup, find_packages

setup(
    name="blackboxaudit",
    version="0.1.0",
    description="A simple package for black box auditing. WIP",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mohammed Rizwan Kumte",
    author_email="NA20B040@smail.iitm.ac.in",
    url="https://github.com/yourusername/your_project",
    packages=find_packages(),  # Automatically find package directories
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
    "torch>=2.4.0",        
    "opacus>=1.5.2",
    "scipy>=1.5.0"   
    ],
)
