
import re
from datetime import date
from pathlib import Path

def set_directory(
        base_dir: str,
        prefix: str = "version",
        with_time: bool = False,
    ):
    base_dir = Path(base_dir)

    if prefix:
        max_version = _find_max_version(base_dir, prefix)
        new_version = max_version + 1
        if with_time:
            today = date.today().strftime("%Y%m%d")
            dir_name = f"{prefix}{new_version}_{today}"
        else:
            dir_name = f"{prefix}{new_version}"
        base_dir = base_dir / dir_name

    base_dir.mkdir(parents=True, exist_ok=True)

    return base_dir


def _find_max_version(base_dir: Path, prefix: str):
    if not base_dir.exists():
        return 0

    pattern = re.compile(r"{}(\d+).*".format(prefix))
    max_version = 0

    for path in base_dir.iterdir():
        if path.is_dir():
            match = pattern.match(path.name)
            if match:
                version = int(match.group(1))
                max_version = max(max_version, version)

    return max_version
