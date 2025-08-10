import os
import re
from pathlib import Path
from typing import Tuple, List, Dict
import sys

class MarkdownProcessor:
    def __init__(self):
        self.processed_files = []
        self.warnings = []
    
    def remove_references(self, content: str) -> str:
        """Remove everything from the last occurrence of 'REFERENCE' or 'Reference' onwards."""
        # Find all occurrences of REFERENCE or Reference (case sensitive)
        ref_pattern = r'(REFERENCE|Reference)'
        matches = list(re.finditer(ref_pattern, content))
        
        if matches:
            # Get the position of the last match
            last_match = matches[-1]
            # Find the start of the line containing the reference
            lines_before = content[:last_match.start()].split('\n')
            # Keep everything up to (but not including) the line with the reference
            content = '\n'.join(lines_before[:-1]) if len(lines_before) > 1 else ''
            content = content.rstrip()
        
        return content
    
    def remove_image_lines(self, content: str) -> str:
        """Remove lines that start with ![Image]."""
        lines = content.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Skip lines that start with ![Image]
            if not line.strip().startswith('![Image]'):
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def extract_figure_captions(self, content: str) -> Dict[str, str]:
        """Extract figure captions that start with 'Figure xxx'."""
        captions = {}
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Match lines starting with "Figure" followed by number/identifier
            figure_match = re.match(r'^Figure\s+(\d+[a-zA-Z]?|\w+)', line, re.IGNORECASE)
            
            if figure_match and len(line.split()) >= 10:  # Skip lines with less than 10 words
                figure_id = figure_match.group(1)
                captions[f"figure_{figure_id}"] = line
        
        return captions
    
    def extract_abstract(self, content: str) -> str:
        """Extract abstract content."""
        lines = content.split('\n')
        abstract_content = []
        capturing = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for dedicated abstract title (like ## Abstract:, # Abstract, etc.)
            if re.match(r'^#+\s*Abstract:?\s*$', line_stripped, re.IGNORECASE):
                capturing = True
                continue
            
            # Check for lines starting with "Abstract"
            elif re.match(r'^Abstract\s+', line_stripped, re.IGNORECASE):
                if len(line_stripped.split()) >= 10:  # Skip if less than 10 words
                    abstract_content.append(line_stripped)
                capturing = True
                continue
            
            # If we're capturing and hit another section header, stop
            elif capturing and re.match(r'^#+\s+', line_stripped):
                break
            
            # If we're capturing, add the line
            elif capturing and line_stripped:
                if len(line_stripped.split()) >= 10:  # Skip if less than 10 words
                    abstract_content.append(line_stripped)
        
        return '\n'.join(abstract_content) if abstract_content else ""
    
    def get_markdown_files(self, input_folder: str) -> List[Path]:
        """Get all markdown files from the input folder."""
        input_path = Path(input_folder)
        
        if not input_path.exists():
            raise ValueError(f"Input folder '{input_folder}' does not exist.")
        
        if not input_path.is_dir():
            raise ValueError(f"'{input_folder}' is not a directory.")
        
        # Find all markdown files
        markdown_files = []
        for ext in ['.md', '.markdown']:
            markdown_files.extend(input_path.glob(f'*{ext}'))
            markdown_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        return sorted(markdown_files)
    
    def create_output_folder(self, output_folder: str) -> Path:
        """Create output folder if it doesn't exist."""
        output_path = Path(output_folder)
        
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            warning_msg = f"Created output folder '{output_folder}'"
            self.warnings.append(warning_msg)
            print(warning_msg)
        
        return output_path
    
    def save_extracted_content(self, output_folder: Path, file_base_name: str, 
                             captions: Dict[str, str], abstract: str, processed_content: str):
        """Save extracted captions, abstract, and processed markdown to files."""
        
        # Save figure captions with filename prefix to avoid conflicts
        for caption_id, caption_text in captions.items():
            caption_file = output_folder / f"{file_base_name}_{caption_id}.txt"
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption_text)
            print(f"  Saved caption: {caption_file.name}")
        
        # Save abstract if found
        if abstract.strip():
            abstract_file = output_folder / f"{file_base_name}_abstract.txt"
            with open(abstract_file, 'w', encoding='utf-8') as f:
                f.write(abstract)
            print(f"  Saved abstract: {abstract_file.name}")
        
        # Save processed markdown content (without references)
        processed_md_file = output_folder / f"{file_base_name}_processed.md"
        with open(processed_md_file, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        print(f"  Saved processed markdown: {processed_md_file.name}")
    
    def process_file(self, file_path: Path, output_folder: Path) -> bool:
        """Process a single markdown file."""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"Processing: {file_path.name}")
            
            # Step 1: Remove references
            content = self.remove_references(content)
            
            # Step 2: Remove image lines
            content = self.remove_image_lines(content)
            
            # Step 3: Extract content
            captions = self.extract_figure_captions(content)
            abstract = self.extract_abstract(content)
            
            # Step 4: Save extracted content
            file_base_name = file_path.stem
            self.save_extracted_content(output_folder, file_base_name, captions, abstract, content)
            
            self.processed_files.append(str(file_path))
            
            #print(f"  ✅ Found {len(captions)} figure captions")
            #print(f"  ✅ Abstract: {'Found' if abstract.strip() else 'Not found'}")
            print()
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing {file_path.name}: {str(e)}")
            return False
    
    def process_folder(self, input_folder: str, output_folder: str):
        """Process all markdown files in input folder and save to output folder."""
        print("=== Markdown Folder Processing Pipeline ===")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print()
        
        try:
            # Get all markdown files
            markdown_files = self.get_markdown_files(input_folder)
            
            if not markdown_files:
                print(f"No markdown files found in '{input_folder}'")
                return
            
            print(f"Found {len(markdown_files)} markdown file(s)")
            print()
            
            # Create output folder
            output_path = self.create_output_folder(output_folder)
            
            # Process each file
            success_count = 0
            for file_path in markdown_files:
                if self.process_file(file_path, output_path):
                    success_count += 1
            
            # Summary
            print("=== Processing Summary ===")
            print(f"Files processed successfully: {success_count}/{len(markdown_files)}")
            print(f"Output folder: {output_path.absolute()}")
            if self.warnings:
                print(f"Warnings: {len(self.warnings)}")
                for warning in self.warnings:
                    print(f"  - {warning}")
            
            # List output files
            output_files = list(output_path.glob('*'))
            if output_files:
                print(f"\nGenerated {len(output_files)} output file(s):")
                for file in sorted(output_files):
                    print(f"  - {file.name}")
            
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


def main():
    """Main function with command line argument support."""
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        print()
        print("Example:")
        print("  python script.py ./input_documents ./output_processed")
        print()
        print("This will process all markdown files in 'input_documents' folder")
        print("and save the results to 'output_processed' folder")
        return
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    processor = MarkdownProcessor()
    processor.process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()