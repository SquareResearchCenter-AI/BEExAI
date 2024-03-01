import os


def get_path(path, check_dir=False):
    """Get the path of a file or directory"""
    dir_path = path.split("/")[:-1]
    dir_path = "/".join(dir_path)
    path_exists = os.path.exists(dir_path) if check_dir else os.path.exists(path)
    if not path_exists:
        from_parent_exists = (
            os.path.exists("../" + dir_path)
            if check_dir
            else os.path.exists("../" + path)
        )
        if from_parent_exists:
            new_path = "../" + path
        else:
            raise FileNotFoundError(f"Path {path} does not exist")
    else:
        new_path = path
    return new_path


def create_dir(path):
    """Create a directory if it does not exist"""
    if not os.path.exists(path):
        os.mkdir(path)
        return
