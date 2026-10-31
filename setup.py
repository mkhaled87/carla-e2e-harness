#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="carla-e2e",
    version="1.0.0",
    description="Standalone Closed-Loop Evaluator for E2E Autonomous Driving Models in CARLA (0.9.14 - 0.10+)",
    author="CARLA Autonomous Driving & AI Team",
    packages=find_packages(),
    py_modules=[
        "test_e2e_closed_loop",
        "e2e_closed_loop_runner",
        "e2e_artifact_recorder",
        "e2e_display",
        "e2e_tracker",
        "e2e_rigs",
        "e2e_schema",
        "e2e_coordinates",
        "e2e_protocol",
        "e2e_model_server",
        "e2e_ros2_bridge",
    ],
    install_requires=[
        "numpy>=1.22.0",
        "pygame>=2.1.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "pillow>=9.0.0",
        "opencv-python>=4.6.0",
    ],
    entry_points={
        "console_scripts": [
            "carla-e2e=test_e2e_closed_loop:main",
            "carla-e2e-server=e2e_model_server:main",
        ],
    },
    python_requires=">=3.10",
)

