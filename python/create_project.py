"""
Project Structure Generator - Creates project directory structures from templates or custom input.
Author: [Your Name]
Version: 1.0.0
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import os
import re
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple, Set
import logging
import shutil
from datetime import datetime
import threading
from queue import Queue, Empty


class StructureParser:
    """Handles parsing of project structure text into dictionary format."""
    
    @staticmethod
    def clean_line(line: str) -> str:
        """Clean a line of structure text."""
        return re.sub(r'[├└─│]', '', line.strip().rstrip('/'))

    @staticmethod
    def get_indent_level(line: str) -> int:
        """Calculate the indent level of a line."""
        return len(line) - len(line.lstrip())

    def _is_directory(self, line: str) -> bool:
        """Check if a line represents a directory."""
        return '.' not in line

    def _update_path(self, current_path: List[str], indent: int, line: str, 
                    current_indent: int) -> Tuple[List[str], int]:
        """Update the current path based on indent level."""
        while indent < current_indent and current_path:
            current_path.pop()
            current_indent -= 2

        if line:
            current_path = current_path[:indent//2]
            if self._is_directory(line):
                current_path.append(line)

        return current_path, indent

    def parse(self, structure_text: str) -> Tuple[str, Dict[str, Union[str, dict]]]:
        """Parse the text-based structure into a dictionary."""
        lines = structure_text.strip().split('\n')
        if not lines:
            raise ValueError("Empty structure")
            
        root_dir = self.clean_line(lines[0]).rstrip('/')
        if not root_dir:
            raise ValueError("Invalid root directory name")
        
        structure: Dict[str, Union[str, dict]] = {}
        current_path: List[str] = []
        current_indent = 0
        
        for line in lines[1:]:
            if not line.strip():
                continue
                
            indent = self.get_indent_level(line)
            clean_line = self.clean_line(line)
            
            if clean_line:
                current_path, current_indent = self._update_path(
                    current_path, indent, clean_line, current_indent
                )
                self._add_to_structure(structure, current_path, clean_line)
        
        return root_dir, structure

    def _add_to_structure(self, structure: Dict[str, Union[str, dict]], 
                         path: List[str], item: str) -> None:
        """Add an item to the structure dictionary at the specified path."""
        current = structure
        
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]  # type: ignore

        if self._is_directory(item):
            if item not in current:
                current[item] = {}
        else:
            current[item] = ''


class ProjectGeneratorApp:
    def __init__(self):
        self.setup_logging()
        self.root = tk.Tk()
        self.root.title("Project Structure Generator")
        self.root.geometry("1000x700")
        
        self.templates = {
            "Python Package": self.get_python_template(),
            "Web Application": self.get_web_template(),
            "Custom": ""
        }
        
        self.message_queue: Queue = Queue()
        self.structure_parser = StructureParser()
        self.last_created_paths: Set[Path] = set()
        self.setup_ui()
        self.check_message_queue()

    def setup_ui(self) -> None:
        """Setup the user interface."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        ttk.Label(main_frame, text="Project Type:").grid(row=0, column=0, sticky="w", pady=5)
        self.project_type = ttk.Combobox(main_frame, values=list(self.templates.keys()), 
                                       state="readonly")
        self.project_type.grid(row=0, column=1, sticky="ew", pady=5)
        self.project_type.current(0)
        self.project_type.bind('<<ComboboxSelected>>', self.on_template_change)
        
        ttk.Label(main_frame, text="Project Structure:").grid(row=1, column=0, sticky="w", pady=5)
        self.structure_text = scrolledtext.ScrolledText(main_frame, height=20, width=80)
        self.structure_text.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=5)
        
        path_frame = ttk.Frame(main_frame)
        path_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(path_frame, text="Project Path:").pack(side="left", padx=5)
        self.path_entry = ttk.Entry(path_frame)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_path).pack(side="left", padx=5)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        self.generate_btn = ttk.Button(button_frame, text="Generate Project", 
                                     command=self.generate_project)
        self.generate_btn.pack(side="left", padx=5)
        
        self.undo_btn = ttk.Button(button_frame, text="Undo Last Generation", 
                                  command=self.undo_last_generation, state="disabled")
        self.undo_btn.pack(side="left", padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, mode='determinate', 
                                      variable=self.progress_var)
        self.progress.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        status_bar.grid(row=6, column=0, columnspan=2, sticky="ew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def setup_logging(self) -> None:
        """Configure logging for the application."""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"project_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        except OSError as e:
            messagebox.showerror("Logging Error", f"Failed to setup logging: {str(e)}")
            raise

    @staticmethod
    def get_python_template() -> str:
        """Return Python project template."""
        return """project_name/
├── __init__.py
├── setup.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── core.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
└── docs/
    └── README.md"""

    @staticmethod
    def get_web_template() -> str:
        """Return web project template."""
        return """webapp/
├── src/
│   ├── components/
│   │   └── __init__.py
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/
├── tests/
├── config.py
└── README.md"""

    def browse_path(self) -> None:
        """Open directory browser dialog."""
        path = filedialog.askdirectory()
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

    def on_template_change(self, event: Optional[tk.Event] = None) -> None:
        """Handle template selection change."""
        template = self.templates.get(self.project_type.get(), "")
        self.structure_text.delete('1.0', tk.END)
        self.structure_text.insert('1.0', template)

    def check_message_queue(self) -> None:
        """Check for messages from worker threads."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.status_var.set(message)
                self.progress_var.set(self.progress_var.get() + 10)
        except Empty:
            pass
        finally:
            self.root.after(100, self.check_message_queue)

    def _create_file(self, path: Path, content: str) -> bool:
        """Create a file at the specified path with given content."""
        try:
            path.parent.mkdir(exist_ok=True, parents=True)
            self.last_created_paths.add(path.parent)
            path.write_text(content or '')
            self.last_created_paths.add(path)
            self.message_queue.put(f"Created file: {path}")
            return True
        except OSError as e:
            self.logger.error(f"Error creating file {path}: {e}")
            return False

    def _create_directory(self, path: Path) -> bool:
        """Create a directory at the specified path."""
        try:
            path.mkdir(exist_ok=True, parents=True)
            self.last_created_paths.add(path)
            self.message_queue.put(f"Created directory: {path}")
            return True
        except OSError as e:
            self.logger.error(f"Error creating directory {path}: {e}")
            return False

    def create_project_structure(self, base_path: str,
                               structure: Dict[str, Union[str, dict]],
                               root_dir_name: str) -> bool:
        """Create the project structure from the parsed dictionary."""
        try:
            # Create the root project directory
            root_dir = Path(base_path) / root_dir_name
            if not self._create_directory(root_dir):
                return False

            def process_directory(current_dir: Path, items: Dict[str, Union[str, dict]]) -> bool:
                """Process a directory and its contents recursively."""
                try:
                    for name, content in items.items():
                        current_path = current_dir / name
                        
                        if isinstance(content, dict):
                            # This is a directory
                            if not self._create_directory(current_path):
                                return False
                            # Recursively process the directory's contents
                            if not process_directory(current_path, content):
                                return False
                        else:
                            # This is a file
                            if not self._create_file(current_path, content):
                                return False
                    return True
                except Exception as e:
                    self.logger.error(f"Error processing directory {current_dir}: {e}")
                    return False

            return process_directory(root_dir, structure)

        except OSError as e:
            self.logger.error(f"Error in create_project_structure: {e}")
            return False

    def _validate_input(self) -> bool:
        """Validate user input."""
        project_path = self.path_entry.get().strip()
        if not project_path:
            messagebox.showerror("Error", "Please select a project path")
            return False
            
        structure = self.structure_text.get('1.0', tk.END).strip()
        if not structure:
            messagebox.showerror("Error", "Please provide project structure")
            return False
            
        return True

    def _remove_path(self, path: Path) -> bool:
        """Remove a file or directory if empty."""
        try:
            if path.is_file():
                path.unlink()
                return True
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                return True
            return False
        except OSError as e:
            self.logger.warning(f"Could not remove {path}: {e}")
            return False

    def undo_last_generation(self) -> None:
        """Undo the last project generation."""
        self.undo_btn.config(state="disabled")
        self.generate_btn.config(state="disabled")
        self.status_var.set("Undoing last generation...")
        self.progress_var.set(0)

        def undo() -> None:
            try:
                paths_to_remove = sorted(
                    self.last_created_paths,
                    key=lambda x: len(str(x).split(os.sep)),
                    reverse=True
                )
                total_paths = len(paths_to_remove)

                for i, path in enumerate(paths_to_remove):
                    self._remove_path(path)
                    self.progress_var.set((i + 1) / total_paths * 100)

                self.last_created_paths.clear()
                self.message_queue.put("Successfully undid last generation")

            except Exception as e:
                self.logger.error(f"Error during undo: {e}")
                self.message_queue.put(f"Error during undo: {str(e)}")
            finally:
                self.root.after(0, lambda: self.generate_btn.config(state="normal"))

        threading.Thread(target=undo, daemon=True).start()

    def generate_project(self) -> None:
        """Handle project generation."""
        if not self._validate_input():
            return

        self.generate_btn.config(state="disabled")
        self.undo_btn.config(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Generating project...")
        self.last_created_paths.clear()

        def generate() -> None:
            try:
                project_path = self.path_entry.get().strip()
                structure_text = self.structure_text.get('1.0', tk.END)
                
                root_dir_name, structure = self.structure_parser.parse(structure_text)
                
                success = self.create_project_structure(project_path, structure, root_dir_name)
                
                if success:
                    self.message_queue.put("Project generated successfully!")
                    self.progress_var.set(100)
                    self.root.after(0, lambda: self.undo_btn.config(state="normal"))
                else:
                    self.message_queue.put("Failed to generate project")
                    
            except Exception as e:
                self.logger.error(f"Error in generate thread: {e}")
                self.message_queue.put(f"Error: {str(e)}")
            finally:
                self.root.after(0, lambda: self.generate_btn.config(state="normal"))

        threading.Thread(target=generate, daemon=True).start()

    def run(self) -> None:
        """Start the application."""
        try:
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise


def main() -> None:
    """Main entry point of the application."""
    try:
        app = ProjectGeneratorApp()
        app.run()
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        messagebox.showerror("Error", f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()