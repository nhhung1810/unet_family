import fnmatch
import os

TEXT_LABEL = [
    "001", 
    "002", 
    "003", 
    "004",
    "005", 
    "006", 
    "007", 
    "008",
    "009", 
    "010"
    ]

def scan_dir(root, pattern):
    """Scan directory and find file that match pattern

    Args:
        root (str): path of directory to begin scanning
        pattern (str): pattern to filter for

    Yields:
        str: Full path to the file
    """
    for dirpath, _, files in os.walk(root):
        files = fnmatch.filter(files, pattern)
        if len(files) == 0:
            continue
        for filename in files:
            yield os.path.join(dirpath, filename)

def assert_data_path(image_dir, seg_dir):
    image_list = list(scan_dir(image_dir, "*.png"))
    for path in image_list:
        base_name, ext = os.path.splitext(os.path.basename(path))
        mask_name = f"{base_name}_seg0.{ext}"
        mask_path = os.path.join(seg_dir, mask_name)
        assert os.path.exists(mask_path), f"{mask_path} doesn't exist -> Check your data"
    pass

if __name__ == "__main__":
    image_dir = "./data/leedsbutterfly/images"
    seg_dir = "./data/leedsbutterfly/segmentations"
    assert_data_path(image_dir, seg_dir)
    image_list = list(scan_dir(image_dir, "*.png"))
    pass