from setuptools import setup, find_packages

setup(
    name="sortwise",
    version="0.1.0",
    description="智能垃圾分类系统",
    author="SortWise Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "efficientnet-pytorch>=0.7.1",
        "flask>=2.0.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.1",
        "tqdm>=4.62.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "jupyter>=1.0.0",
        ]
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "sortwise-train=scripts.train:main",
            "sortwise-evaluate=scripts.evaluate:main",
            "sortwise-predict=scripts.predict:main",
        ]
    },
    include_package_data=True,
    package_data={
        "sortwise": [
            "config.yaml",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 