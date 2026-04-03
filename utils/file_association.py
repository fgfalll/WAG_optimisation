"""
File association utility for CO₂ EOR Optimizer.
Handles Windows registry operations for .phd file association.
"""
import logging
import sys
import winreg
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class FileAssociationManager:
    """Manages file associations for .phd files on Windows."""
    
    def __init__(self, app_name: str = "CO2EOROptimizer"):
        self.app_name = app_name
        self.prog_id = f"{app_name}.phd"
        self.registry_path = f"Software\\Classes\\{self.prog_id}"
        
    def associate_phd_files(self, executable_path: Path, icon_path: Optional[Path] = None) -> bool:
        """
        Associate .phd files with this application in Windows registry.
        
        Args:
            executable_path: Path to the application executable
            icon_path: Optional path to the icon file (.ico)
            
        Returns:
            True if association was successful, False otherwise
        """
        if not sys.platform.startswith('win'):
            logger.warning("File associations are only supported on Windows")
            return False
            
        try:
            executable_str = str(executable_path.resolve())
            
            # Create the main program ID key
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, self.registry_path) as key:
                winreg.SetValue(key, "", winreg.REG_SZ, "CO₂ EOR Project File")
                
                # Set default icon if provided
                if icon_path and icon_path.exists():
                    icon_str = str(icon_path.resolve())
                    with winreg.CreateKey(key, "DefaultIcon") as icon_key:
                        winreg.SetValue(icon_key, "", winreg.REG_SZ, icon_str)
                
                # Create shell open command
                with winreg.CreateKey(key, "shell\\open\\command") as cmd_key:
                    winreg.SetValue(cmd_key, "", winreg.REG_SZ, f'"{executable_str}" "%1"')
            
            # Associate .phd extension with our program ID
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Software\\Classes\\.phd") as ext_key:
                winreg.SetValue(ext_key, "", winreg.REG_SZ, self.prog_id)
            
            logger.info(f"Successfully associated .phd files with {executable_str}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create file association: {e}")
            return False
    
    def remove_association(self) -> bool:
        """Remove .phd file association from Windows registry."""
        if not sys.platform.startswith('win'):
            return True
            
        try:
            # Remove extension association
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, "Software\\Classes\\.phd")
            except FileNotFoundError:
                pass
            
            # Remove program ID key
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, self.registry_path)
            except FileNotFoundError:
                pass
                
            logger.info("Removed .phd file association")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove file association: {e}")
            return False
    
    def is_associated(self) -> bool:
        """Check if .phd files are currently associated with this application."""
        if not sys.platform.startswith('win'):
            return False
            
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Software\\Classes\\.phd") as key:
                value, _ = winreg.QueryValueEx(key, "")
                return value == self.prog_id
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking association: {e}")
            return False


def setup_file_association(executable_path: Path, icon_path: Optional[Path] = None) -> bool:
    """Convenience function to set up file association."""
    manager = FileAssociationManager()
    return manager.associate_phd_files(executable_path, icon_path)


def remove_file_association() -> bool:
    """Convenience function to remove file association."""
    manager = FileAssociationManager()
    return manager.remove_association()