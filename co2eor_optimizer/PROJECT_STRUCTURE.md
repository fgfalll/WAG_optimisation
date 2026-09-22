# Project Structure Documentation

## Overview
This document describes the structure of the CO2 EOR Optimization project. The project is organized into logical components including core functionality, user interface, analysis tools, and utility modules.

## Root Directory
- `config_manager.py` - Configuration management module
- `data_processor.py` - Data processing utilities
- `main.py` - Main application entry point
- `requirements.txt` - Python dependencies

## Analysis Modules
- `analysis/` - Contains analysis tools
  - `sensitivity_analyzer.py` - Sensitivity analysis implementation
  - `uq_engine.py` - Uncertainty quantification engine
  - `well_analysis.py` - Well-specific analysis tools

## Configuration Files
- `config/` - Configuration files
  - `base_config.json` - Base configuration
  - `economic_scenarios.json` - Economic scenario definitions
  - `recovery_config.json` - Recovery configuration
  - `uq_and_sensitivity.json` - Uncertainty/sensitivity settings

## Core Functionality
- `core/` - Core application logic
  - `data_models.py` - Data models and schemas
  - `eos_models.py` - Equation of State models
  - `optimisation_engine.py` - Optimization algorithms
  - `recovery_models.py` - Recovery models

## Evaluation Modules
- `evaluation/` - Evaluation components
  - `mmp.py` - Minimum miscibility pressure calculations

## Parsers
- `parsers/` - Data parsers
  - `base_parser.py` - Base parser interface
  - `eclipse_parser.py` - Eclipse data format parser
  - `las_parser.py` - LAS file parser
  - `validation.py` - Data validation utilities

## User Interface
- `ui/` - UI components
  - Main widgets: `ai_assistant_widget.py`, `config_widget.py`, `data_management_widget.py`, etc.
  - `dialogs/` - Dialog components
    - `parameter_help_dialog.py`, `plot_customization_dialog.py`, etc.
  - `icons/` - Application icons
    - `spiner.gif`
  - `widgets/` - Custom widgets
    - `depth_profile_dialog.py`, `file_list_widget.py`, `pvt_editor_dialog.py`, etc.
  - `workers/` - Background workers
    - `ai_query_worker.py`, `data_processing_worker.py`, `optimization_worker.py`, etc.

## Utilities
- `utils/` - Utility modules
  - `grdecl_writer.py` - GRDECL file writer
  - `i18n_manager.py` - Internationalization support
  - `project_file_handler.py` - Project file management
  - `report_generator.py` - Report generation
  - `units_manager.py` - Unit conversion and management