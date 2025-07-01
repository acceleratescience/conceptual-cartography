# Streamlit Applications

This directory contains Streamlit applications for interactive visualization of conceptual cartography results.

## Apps

### `landscape_viewer.py`
Interactive visualization tool for exploring conceptual landscapes and metrics across different transformer layers.

**Usage:**
```bash
# Via the CLI script (recommended)
poetry run visualize --config configs/your-config.yaml

# Direct usage
streamlit run streamlit/landscape_viewer.py -- path/to/results target_word
```

**Features:**
- Layer navigation with Previous/Next buttons
- Direct layer selection via dropdown
- Interactive 3D landscape plots
- Metrics display (MEV, similarities, etc.)
- File availability status

## Architecture

The Streamlit apps are kept separate from the main package (`src/`) to avoid import complications and maintain clean separation of concerns:

- **Package code** (`src/`): Core functionality, data models, computation
- **CLI scripts** (`scripts/`): Command-line interfaces
- **Apps** (`streamlit/`): Interactive web applications

This structure allows the Streamlit apps to cleanly import from the package while being run externally.

## Adding New Apps

When adding new Streamlit applications:

1. Create the app file in this directory
2. Use the import pattern shown in `landscape_viewer.py`:
   ```python
   # Add project root to path for clean imports
   project_root = Path(__file__).parent.parent
   if str(project_root) not in sys.path:
       sys.path.insert(0, str(project_root))
   
   # Import from package
   from src.module import Class
   ```
3. Add a CLI script in `scripts/` if needed
4. Document usage here 