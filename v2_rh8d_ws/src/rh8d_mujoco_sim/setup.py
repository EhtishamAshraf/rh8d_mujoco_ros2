from setuptools import setup

package_name = 'rh8d_mujoco_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='labauto',
    maintainer_email='ehtishamashraf67@gmail.com',
    description='ROS2 MuJoCo simulation for RH8D hand',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rh8d_mujoco_node = rh8d_mujoco_sim.rh8d_mujoco_node:main',
            'rh8d_hand_test = rh8d_mujoco_sim.rh8d_hand_test:main',
        ],
    },
)