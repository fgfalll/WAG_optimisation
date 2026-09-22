import logging
import yaml
import markdown
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# --- DYNAMIC PATH RESOLUTION ---
# This is the key to making the package self-contained.
# It resolves paths relative to THIS file's parent directory, which is `co2eor_optimizer`.
PACKAGE_DIR = Path(__file__).parent.resolve()

class HelpManager:
    """
    A singleton to manage loading and accessing help content from within the package.
    It reads a YAML manifest from a 'config' directory and loads Markdown files
    from a 'docs' directory, both located as siblings to this file.
    """
    _instance = None
    _help_data = {}
    _page_cache = {}

    # --- UPDATED PATHS ---
    # The paths are now built dynamically from the package directory location.
    # This will work correctly whether run from source or as a packaged application.
    _manifest_path = PACKAGE_DIR / "config" / "help_content.yaml"
    _base_docs_path = PACKAGE_DIR / "docs" / "help"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_manifest()
        return cls._instance

    def load_manifest(self):
        """Loads the YAML manifest file into memory. This is called once."""
        if not self._manifest_path.exists():
            logger.error(f"Help manifest not found at '{self._manifest_path}'. Ensure the 'config/help_content.yaml' file exists within the 'co2eor_optimizer' package. Help will be unavailable.")
            self._help_data = {}
            return

        try:
            with open(self._manifest_path, 'r', encoding='utf-8') as f:
                self._help_data = yaml.safe_load(f)
            logger.info(f"Help manifest loaded successfully from '{self._manifest_path}'.")
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load or parse help manifest: {e}", exc_info=True)
            self._help_data = {}

    def get_short_help(self, key: str) -> str:
        """Returns the short tooltip text from the manifest for a given key."""
        try:
            parts = key.split('.')
            scope = parts[0]
            param_key = '.'.join(parts[1:])
            return self._help_data[scope][param_key].get('short', "No description available.")
        except (KeyError, IndexError, AttributeError):
            logger.debug(f"Could not find short help for key: {key}")
            return "No description available."

    def get_page_content_for_key(self, key: str) -> Optional[Tuple[str, str, str]]:
        """Gets the full HTML content for a page and the specific anchor for the given key."""
        try:
            parts = key.split('.')
            scope = parts[0]
            param_key = '.'.join(parts[1:])

            page_info = self._help_data.get(scope)
            if not page_info: raise KeyError(f"Scope '{scope}' not found in manifest.")
            param_info = page_info.get(param_key)
            if not param_info: raise KeyError(f"Key '{param_key}' not in scope '{scope}'.")

            md_filename = page_info.get('_file')
            anchor_id = param_info.get('anchor')
            page_title = scope.replace("Widget", " Help")

            if not md_filename or anchor_id is None:
                logger.warning(f"Incomplete help info for '{key}'. Missing _file or anchor.")
                return None

            if md_filename in self._page_cache:
                html_content = self._page_cache[md_filename]
            else:
                full_md_path = self._base_docs_path / md_filename
                md_content = full_md_path.read_text(encoding='utf-8')
                html_content = markdown.markdown(md_content, extensions=['fenced_code', 'tables'])
                self._page_cache[md_filename] = html_content
                logger.info(f"Cached help page from '{full_md_path}'.")

            return (page_title, html_content, anchor_id)

        except FileNotFoundError as e:
            logger.error(f"Help file specified not found: {e.filename}")
            html = f"<h1>Error</h1><p>File not found: <code>{e.filename}</code></p>"
            return ("File Not Found", html, "")
        except (KeyError, IndexError, AttributeError) as e:
            logger.warning(f"No valid help entry for key '{key}'. Details: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching help for key '{key}': {e}", exc_info=True)
            return None