from setuptools import find_packages, setup


package_name = "lidar_detection_runtime"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ngy-Gergo",
    maintainer_email="gergonagy05@gmail.com",
    description="Production vehicle inference and live sensor integration.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
