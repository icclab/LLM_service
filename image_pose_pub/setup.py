from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'image_pose_pub'

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
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_image_pub = image_pose_pub.pose_image_pub:main',
            'image_decompressor = image_pose_pub.image_decompressor:main',
        ],
    },
)
