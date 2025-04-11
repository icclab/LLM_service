from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'liquid_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lei',
    maintainer_email='aries94leifu@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'roboflow = liquid_detection.roboflow_water:main',
            'water = liquid_detection.water:main',
            'pixel_water = liquid_detection.roboflow:main',
        ],
    },
)
