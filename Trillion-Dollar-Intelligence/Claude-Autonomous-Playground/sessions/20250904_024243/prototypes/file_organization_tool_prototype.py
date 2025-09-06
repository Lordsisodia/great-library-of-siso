#!/usr/bin/env python3
"""
File Organization Tool Prototype
Automatically organizes files by type and date
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import mimetypes

class FileOrganizer:
    def __init__(self, source_dir: str, organized_dir: str):
        self.source_dir = Path(source_dir)
        self.organized_dir = Path(organized_dir)
        self.organized_dir.mkdir(exist_ok=True)
        
        self.type_mapping = {
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.md'],
            'spreadsheets': ['.xls', '.xlsx', '.csv'],
            'code': ['.py', '.js', '.html', '.css', '.json', '.yaml', '.yml'],
            'archives': ['.zip', '.rar', '.tar', '.gz'],
            'videos': ['.mp4', '.avi', '.mov', '.mkv'],
            'audio': ['.mp3', '.wav', '.flac', '.aac']
        }
    
    def organize_files(self):
        """Organize all files in source directory"""
        moved_files = []
        
        for file_path in self.source_dir.iterdir():
            if file_path.is_file():
                file_type = self._get_file_type(file_path)
                dest_dir = self.organized_dir / file_type
                dest_dir.mkdir(exist_ok=True)
                
                dest_path = dest_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                
                moved_files.append({
                    'original': str(file_path),
                    'new': str(dest_path),
                    'type': file_type
                })
        
        return moved_files
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type based on extension"""
        extension = file_path.suffix.lower()
        
        for file_type, extensions in self.type_mapping.items():
            if extension in extensions:
                return file_type
        
        return 'other'

if __name__ == "__main__":
    organizer = FileOrganizer("./test_files", "./organized")
    result = organizer.organize_files()
    print(f"Organized {len(result)} files")
