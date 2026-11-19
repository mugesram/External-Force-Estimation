from setuptools import setup

setup(
    name="efe_ros",
    version="0.1.0",
    packages=["efe_ros"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/efe_ros"]),
                ("share/efe_ros", ["package.xml"])],
    install_requires=["setuptools"],
    entry_points={"console_scripts": [
        "offboard_node = efe_ros.offboard_node:main",
    ]},
)
