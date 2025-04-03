from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name="cheap-proteins",
        version=1.0.0,
        author="Amy X. Lu",
        license="MIT",
        author_email="amyxlu@berkeley.edu",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
    )
