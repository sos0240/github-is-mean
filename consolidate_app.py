import subprocess
import os

# Function to extract a file from a feature branch without switching branches


def extract_file_from_branch(branch_name, file_path):
    """Extract a file from a git branch without switching branches.

    Parameters
    ----------
    branch_name : str
        The branch to extract from (e.g. 'feature/my-feature').
    file_path : str
        Path to the file, relative to the repository root.
        ``git show`` requires repo-relative paths, not absolute paths.
    """
    # git show requires repo-relative paths -- reject absolute paths
    # because git show branch:/absolute/path will always fail
    if os.path.isabs(file_path):
        raise ValueError(
            "file_path must be relative to the repository root, not absolute. "
            f"Got: {file_path}"
        )

    try:
        command = ["git", "show", f"{branch_name}:{file_path}"]
        output_name = os.path.basename(file_path)
        with open(output_name, "wb") as f:
            subprocess.run(command, stdout=f, check=True)
        print(f"Successfully extracted {file_path} from {branch_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting file: {e}")


# Example usage:
# extract_file_from_branch('feature/my-feature', 'path/to/my_file.txt')
