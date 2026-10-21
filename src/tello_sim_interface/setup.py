from setuptools import find_packages, setup

package_name = 'tello_sim_interface'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='haohao9555',
    maintainer_email='243935570+haohao9555@users.noreply.github.com',
    description='Tello 仿真接口节点',
    license='MIT',
    entry_points={
        'console_scripts': [
            'interface_node = tello_sim_interface.interface_node:main',
        ],
    },
)
