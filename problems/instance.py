def layout_paths(layout: str, class_name: str | None) -> list[Path]:
    """一个文件就返回它；一个目录配上 class 就返回这一组文件。"""

def read_layout(path: Path) -> dict:
    """读一个文件，返回 num_bays、num_rows、max_tiers、num_containers 和初始堆场。"""