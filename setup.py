from setuptools import find_packages, setup


with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='predicting-loan',
      version="0.0.4",
      description="Predict loan default",
      license="MIT",
      author="Le Wagon",
      author_email="contact@lewagon.org",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
