import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages()
print(packages)

package_data = {'fastplm.files': ['fastplm/files/*.json', 'fastplm/files/*.txt']}

setuptools.setup(
    name="fastplm", # Replace with your own username
    version="0.0.1",
    author="Weijie Liu",
    author_email="dataliu@pku.edu.cn",
    description="The FastPLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autoliuweijie/FastPLM",
    packages=packages,
    package_data=package_data,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
    install_requires=[
        'torch>=1.0.0',
        ]
)
