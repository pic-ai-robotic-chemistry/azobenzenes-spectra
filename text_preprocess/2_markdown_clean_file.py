#!/usr/bin/env python3
"""
Base64 Image Remover for Markdown Files

This script removes base64-encoded images from markdown files,
deleting the image references and updating the markdown in place.
"""

import re
import argparse
from pathlib import Path

def remove_base64_images(markdown_content: str) -> str:
    """
    Remove base64-encoded images from markdown content.

    Args:
        markdown_content: The markdown file content as a string

    Returns:
        Modified markdown content with base64 images removed
    """
    # Pattern to match base64 images in markdown
    pattern = r'!\[[^\]]*\]\(data:image/[^;]+;base64,[^)]+\)'
    # Remove all matches
    return re.sub(pattern, '', markdown_content)

def process_markdown_file(file_path: str) -> None:
    """
    Process a markdown file to remove base64 images.

    Args:
        file_path: Path to the markdown file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Read markdown file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Remove base64 images
    modified_content = remove_base64_images(content)

    # Write modified content back to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"Updated markdown file: {file_path}")
    except Exception as e:
        print(f"Error writing updated file: {e}")
        return

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove base64-encoded images from markdown files"
    )
    parser.add_argument(
        "file_path",
        help="Path to the markdown file to process"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the original file before modifying"
    )

    args = parser.parse_args()

    # Create backup if requested
    if args.backup:
        file_path = Path(args.file_path)
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"Created backup: {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")

    # Process the file
    process_markdown_file(args.file_path)

if __name__ == "__main__":
    main()