from pathlib import Path

import medrisk_health_analytics


def test_version_is_available():
    assert medrisk_health_analytics.__version__


def test_package_declares_typing_support():
    package_dir = Path(medrisk_health_analytics.__file__).parent

    assert (package_dir / "py.typed").is_file()
