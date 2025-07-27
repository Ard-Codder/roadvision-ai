from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="roadvision-ai",
    version="1.0.0",
    author="RoadVision AI Team",
    author_email="contact@roadvision-ai.com",
    description="Интегрированная система детекции дороги и объектов с использованием YOLOv8 и OpenVINO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ard-Codder/roadvision-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Computer Vision",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "jetson": [
            "numpy>=1.21.0",
            "opencv-python>=4.8.0",
            "torch>=2.0.0",
            "ultralytics>=8.0.0",
            "openvino>=2023.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "roadvision-ai=integrated_road_detection:main",
            "process-videos=process_videos:main",
            "test-system=test_system:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.bat"],
    },
    keywords="computer-vision, road-detection, object-detection, yolo, openvino, segmentation, autonomous-driving, ai, artificial-intelligence",
    project_urls={
        "Bug Reports": "https://github.com/Ard-Codder/roadvision-ai/issues",
        "Source": "https://github.com/Ard-Codder/roadvision-ai",
        "Documentation": "https://github.com/Ard-Codder/roadvision-ai#readme",
        "Jetson Orin": "https://github.com/Ard-Codder/roadvision-ai/tree/main/jetson_orin",
        "Contributing": "https://github.com/Ard-Codder/roadvision-ai/blob/main/docs/project/CONTRIBUTING.md",
        "Changelog": "https://github.com/Ard-Codder/roadvision-ai/blob/main/docs/project/CHANGELOG.md",
    },
) 