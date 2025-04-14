from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PyFARM",  # Package name
    version="0.1.0",  # Package version
    author="Farm",  # Author's name
    author_email="farm@farm.com",  # Author's email address
    description="A Python package for time series feature alignment and distance computation with the implementation of the Forward Angular Relevance Metric.",
    long_description=long_description,  # Long description from README
    long_description_content_type="text/markdown",  # Specify the content type for README
    url="https://github.com/avilarenan/PyFARM",  # Replace with your actual repository URL
    packages=find_packages(),  # Automatically find all packages in the directory
    classifiers=[
        "Programming Language :: Python :: 3",  # Ensure compatibility with Python 3
        "License :: OSI Approved :: MIT License",  # License type
        "Operating System :: OS Independent",  # OS independent
    ],
    python_requires='>=3.6',  # Specify minimum Python version
    install_requires=[  # List any dependencies your package needs
        "numpy>=1.21.0",   # For numerical operations
        "scipy>=1.7.0",     # For advanced numerical calculations (e.g., statistics, signal processing)
        "pandas>=1.3.0",    # For handling time series data (if used)
        "matplotlib>=3.4.0", # Optional: For visualization
        "scikit-learn>=0.24.0" # Optional: If you plan on expanding to machine learning
    ],
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    entry_points={  # Entry points for command-line tools if you have any
        # "console_scripts": [
        #     "your-command=your_module:main_function",
        # ],
    },
)
