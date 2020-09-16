from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
      name="SoftAdapt",
      version="0.0.2",
      author="A. Ali Heydari",
      author_email="aliheydari@ucdavis.edu",
      description="SoftAdapt: Techniques for Adaptive Loss Weighting of Neural Networks with Multi-Part Loss Functions",
      long_description=readme,
      long_description_content_type="text/markdown",
      license="MIT",
      url="https://github.com/dr-aheydari/SoftAdapt",
      download_url="https://github.com/dr-aheydari/SoftAdapt",
      packages=find_packages(),
      install_requires=[
                        'tqdm',
                        'numpy',
                        ],
      classifiers=[
                   "Development Status :: 4 - Beta",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT Software License",
                   "Programming Language :: Python :: 3.6",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence"
                   ],
      keywords="Adaptive-Weighting Multi-Task-Nerual-Networks Optimization Gradient-Descent-Weighting"
      )
