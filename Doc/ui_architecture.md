All right, let's outline a detailed structure for both the UI (PyQt6) and the overall Python project. This will serve as a blueprint for development.

**I. Overall Project Directory Structure (Python Project):**

```
co2_eor_suite/
├── main.py                     # Main application entry point (PyQt6 app initialization)
├── ui/                         # PyQt6 UI specific files
│   ├── main_window.py          # Defines the QMainWindow, tab structure, core UI logic
│   ├── overview_page.py        # Widget/Logic for the initial overview screen
│   ├── data_management_widget.py # Widget/Logic for Tab 1 (Data Input & Management)
│   ├── config_widget.py        # Widget/Logic for Tab 2 (Configuration)
│   ├── mmp_well_analysis_widget.py # Widget for Tab 3
│   ├── optimization_widget.py    # Widget for Tab 4
│   ├── sensitivity_widget.py     # Widget for Tab 5
│   ├── uq_widget.py              # Widget for Tab 6
│   ├── ai_assistant_widget.py    # Widget for Tab 7 (AI Integration)
│   ├── logging_widget.py         # Widget for Tab 8 (Logs)
│   ├── dialogs/                  # Custom dialogs
│   │   ├── parameter_help_dialog.py
│   │   ├── scenario_editor_dialog.py
│   │   ├── plot_customization_dialog.py
│   │   └── report_config_dialog.py
│   ├── widgets/                  # Reusable custom UI components
│   │   ├── file_list_widget.py
│   │   ├── parameter_input_group.py # For dynamic forms with help buttons
│   │   └── pvt_table_editor.py
│   └── workers/                  # QThread worker classes for backend tasks
│       ├── data_processing_worker.py
│       ├── optimization_worker.py
│       ├── sensitivity_analysis_worker.py
│       └── uq_worker.py
│       └── ai_query_worker.py
├── core/                       # Core backend logic (as you currently have)
│   ├── data_models.py
│   ├── optimisation_engine.py
│   ├── recovery_models.py
├── analysis/                   # Analysis engines (as you currently have)
│   ├── sensetivity_analyser.py
│   ├── uq_engine.py
│   ├── well_analysis.py
├── evaluation/                 # Evaluation modules (as you currently have)
│   ├── mmp.py
├── parsers/                    # Data parsers (as you currently have)
│   ├── eclipse_parser.py
│   ├── las_parser.py
├── utils/                      # Utility modules (as you currently have)
│   ├── grdecl_writer.py
│   ├── units_manager.py        # NEW: For unit conversions
│   ├── i18n_manager.py         # NEW: For handling translations setup (conceptual)
│   ├── report_generator.py     # NEW: For generating reports
│   └── project_file_handler.py # NEW: For .tphd file save/load logic
├── config/                     # Configuration files (as you currently have)
│   ├── base_config.json
│   ├── economic_scenarios.json
│   ├── uq_and_sensitivity.json
├── translations/               # For internationalization
│   ├── app_en.qm
│   ├── app_uk.qm
│   ├── app_en.ts               # Source translation files
│   └── app_uk.ts
├── docs/                       # Project documentation
├── .tphd_schema.json           # JSON schema for validating .tphd files
├── config_manager.py           # (Your existing global config manager)
├── data_processor.py           # (Your existing global data processor - maybe move to core or utils)
├── requirements.txt
└── README.md
```

**II. UI Structure (PyQt6 - Detailed Breakdown):**

**`main.py`**
*   Initializes `QApplication`.
*   Handles global settings (e.g., loading `QSettings`, setting up initial `QTranslator`).
*   Instantiates and shows `MainWindow`.
*   Starts the Qt event loop (`app.exec()`).
*   Global logging setup.

**`ui/main_window.py` - `MainWindow(QMainWindow)`**
*   **Attributes (State Management):**
    *   `project_file_path`: Path to the currently open `.tphd` file.
    *   `is_project_modified`: Boolean flag.
    *   `active_unit_system`: String (e.g., "Field", "SI").
    *   `config_manager_instance`: Reference to your global `config_manager`.
    *   `project_file_handler_instance`: Instance of `ProjectFileHandler`.
    *   `units_manager_instance`: Instance of `UnitsManager`.
    *   `report_generator_instance`: Instance of `ReportGenerator`.
    *   `app_settings`: Instance of `QSettings` for persisting app-level prefs.
    *   `current_data_models`: Dictionary holding active instances of `ReservoirData`, `PVTProperties`, `WellData` (list), etc. that are being worked on.
    *   `current_config_dataclasses`: Dictionary holding active instances of `EconomicParameters`, `EORParameters`, etc.
    *   `optimisation_engine_instance`: Lazily initialized when needed.
    *   `sensitivity_analyzer_instance`: Lazily initialized.
    *   `uq_engine_instance`: Lazily initialized.
    *   References to UI widgets (e.g., `self.overview_page_widget`, `self.main_tab_widget`).
*   **Methods:**
    *   `__init__()`: Sets up window properties, creates actions, menus, status bar, calls `_setup_initial_view()`.
    *   `_create_actions()`: Defines `QAction` for menu items (Open, Save, Save As, Quick Start, Settings, Quit, Help, etc.).
    *   `_create_menu_bar()`: Populates the `QMenuBar`.
    *   `_create_status_bar()`: Initializes the `QStatusBar`.
    *   `_setup_initial_view()`: Creates and sets `OverviewPageWidget` as the central widget.
    *   `_setup_main_app_tabs_container()`: Creates the main `QTabWidget` (hidden initially).
    *   `_transition_to_main_app_view()`: Hides overview, shows main tab widget.
    *   `_update_window_title()`: Updates title based on project name and modification status.
    *   `_handle_quick_start()`: Slot for "Quick Start" button.
    *   `_load_project()`: Slot for "Open Project" action.
    *   `_save_project()`: Slot for "Save Project" action.
    *   `_save_project_as()`: Slot for "Save Project As" action.
    *   `_show_settings_dialog()`: Opens a settings/preferences dialog.
    *   `_show_about_dialog()`: Shows an "About" dialog.
    *   `_update_ui_for_unit_system()`: Iterates through relevant UI elements to update labels/conversions.
    *   `closeEvent(event)`: Handles unsaved changes confirmation.
    *   Methods to initialize/pass data to tab widgets.
    *   Methods to receive results from worker threads and update UI/data models.

**`ui/overview_page.py` - `OverviewPageWidget(QWidget)`**
*   **UI Elements:** `QLabel` for title/description, `QPushButton` for "Start New," "Load Project," "Quick Start."
*   **Signals:** Emits signals when buttons are clicked, e.g., `start_new_project_requested = pyqtSignal()`.
*   `MainWindow` connects to these signals.

**`ui/data_management_widget.py` - `DataManagementWidget(QWidget)` (Tab 1)**
*   **UI Elements:**
    *   `QGroupBox` "File Loading":
        *   `QPushButton` "Load LAS", `QPushButton` "Load Reservoir Model".
        *   `FileListViewWidget` (custom widget) for LAS, another for Reservoir files.
        *   `QPushButton` "Process Loaded Files".
    *   `QGroupBox` "Data Review & Manual Input":
        *   `QTabWidget` (sub-tabs): "Well Data", "Reservoir Properties", "Fluid & PVT".
        *   **Well Data Sub-Tab:** `QComboBox` for well selection, `QPushButton` "View/Edit Well Logs" (opens `WellLogEditorDialog`). Prompt for missing well names.
        *   **Reservoir Properties Sub-Tab:** Display grid info. OOIP section with calculator inputs (Area, Thick, Phi, Sw, Bo) and a `QLineEdit` for final OOIP (can be calculated or manual). "?" help for OOIP components.
        *   **Fluid & PVT Sub-Tab:** `PVTTableEditorWidget` (custom widget) or dynamic form for `PVTProperties`. Auto-populate attempt, user override capability. "?" help for each PVT param.
    *   `QTextEdit` for processing logs.
    *   `QPushButton` "Confirm Data & Proceed".
*   **Logic:** Handles file selection, initiates `DataProcessingWorker`, displays results, populates review sections. Manages `current_data_models` being edited.

**`ui/config_widget.py` - `ConfigWidget(QWidget)` (Tab 2)**
*   **UI Elements:**
    *   `QTabWidget` or `QScrollArea` with `QGroupBoxes` for: `EconomicParameters`, `EORParameters`, `OperationalParameters`, `ProfileParameters`, `GeneticAlgorithmParams`, `RecoveryModelKwargsDefaults`.
    *   Inside each group, use `ParameterInputGroupWidget` (custom) or `QFormLayout` to display:
        *   `QLabel` for parameter name.
        *   Appropriate input widget (`QDoubleSpinBox`, `QLineEdit`, `QComboBox`).
        *   `QPushButton("?")` for help.
    *   `QPushButton` "Load Config from File", `QPushButton` "Save Config to File".
    *   `QPushButton` "Reset All to Application Defaults".
*   **Logic:** Populates UI from `MainWindow.current_config_dataclasses` (or `config_manager`). Updates these instances on user input.

**`ui/mmp_well_analysis_widget.py` - `MMPWellAnalysisWidget(QWidget)` (Tab 3)**
*   **UI Elements:**
    *   `QComboBox` to select active well (populated from loaded `WellData`).
    *   `QComboBox` for MMP calculation method.
    *   `QPushButton` "Calculate/Update MMP Profile".
    *   Plotting area (e.g., `QWebEngineView` for Plotly or `PyQtGraph.PlotWidget`).
    *   `QPushButton` "Advanced Log Visualization..." (opens `WellLogEditorDialog` from Tab 1, or a similar one).
*   **Logic:** Triggers `WellAnalysis` in a worker. Displays MMP profile plot.

**`ui/optimization_widget.py` - `OptimizationWidget(QWidget)` (Tab 4)**
*   **UI Elements:**
    *   `QGroupBox` "Setup": Method, Objective selectors. Dynamic area for method-specific GA/Bayesian params (could be a stacked widget).
    *   `QPushButton` "Run Optimization".
    *   `QGroupBox` "Results": `QTableView` for optimized params, `QLabel`s for objective values, RF.
    *   Plotting area for convergence.
    *   `QPushButton` "Plot Parameter Sensitivity (from this optimum)..." (links to functionality in Sensitivity Tab).
*   **Logic:** Initiates `OptimizationWorker`. Displays results and plots.

**`ui/sensitivity_widget.py` - `SensitivityWidget(QWidget)` (Tab 5)**
*   **UI Elements:**
    *   `QTreeView` or `CheckableQListWidget` for selecting parameters across scopes (Econ, EOR, Fluid, Reservoir, Model).
    *   Dynamic panel to set range/values/steps for selected parameter. "?" help.
    *   `QComboBox` for objective(s) to analyze.
    *   `QPushButton` "Run One-Way SA", "Run Two-Way SA", "Run Re-Optimization SA".
    *   `QTabWidget` for results: "Tornado Plot", "Spider Plot", "Data Table", "Contour Plot".
*   **Logic:** Configures and runs `SensitivityAnalysisWorker`. Displays various plots and tables.

**`ui/uq_widget.py` - `UQWidget(QWidget)` (Tab 6)**
*   **UI Elements:**
    *   Similar parameter selection as Sensitivity Tab, but for defining distributions (type, params). "?" help.
    *   `QTableWidget` to manage/edit list of uncertain parameters.
    *   Input for correlation matrix (advanced, maybe a text edit with validation).
    *   `QPushButton` "Run MC (Fixed Strategy)", "Run MC (Opt Under UQ)", "Run PCE".
    *   `QTabWidget` for results: "Distribution Plot (PDF/CDF)", "Statistics Table", "Sobol Indices (PCE)".
*   **Logic:** Configures and runs `UQWorker`. Displays UQ results.

**`ui/ai_assistant_widget.py` - `AIAssistantWidget(QWidget)` (Tab 7)**
*   **UI Elements:**
    *   `QGroupBox` "API Configuration": `QLineEdit` for API keys (masked), `QComboBox` for service (Gemini, OpenAI, OpenRouter), `QLineEdit` for base URL (if not default). `QPushButton` "Save API Config".
    *   `QComboBox` "Select AI Model".
    *   `QTextEdit` for user query/prompt.
    *   `QTextBrowser` for AI response.
    *   `QPushButton` "Send Query to AI".
    *   `QComboBox` or `QListWidget` "Context/Task":
        *   "General Query"
        *   "Explain Parameter (select from Config Tab)"
        *   "Interpret Sensitivity Results (from SA Tab output)"
        *   "Interpret UQ Results (from UQ Tab output)"
        *   "Brainstorm EOR Strategy"
        *   "Draft Report Section"
    *   `QLabel` (prominent) with AI Disclaimer.
*   **Logic:** Manages API keys (securely via `QSettings`). Constructs appropriate prompts based on selected task and current application data. Uses `AIQueryWorker` to make API calls.

**`ui/logging_widget.py` - `LoggingWidget(QWidget)` (Tab 8)**
*   **UI Elements:** `QTextEdit` (read-only) for application logs. `QComboBox` to filter log level display. `QPushButton` "Clear Logs", `QPushButton` "Save Logs to File".
*   **Logic:** Custom `logging.Handler` writes to the `QTextEdit`.

**Custom Dialogs (`ui/dialogs/`)**
*   Each for a specific purpose, e.g., `ParameterHelpDialog(QDialog)` takes a parameter key and displays its pre-defined help text. `ScenarioEditorDialog` allows detailed configuration of a comparative scenario.

**Custom Widgets (`ui/widgets/`)**
*   `FileListViewWidget(QListWidget)`: Shows file path and a status icon/text.
*   `ParameterInputGroupWidget(QWidget)`: A reusable component containing a `QLabel`, an input widget, and a "?" button for a single parameter. Simplifies building config forms.
*   `PVTTableEditorWidget(QWidget)`: A `QTableWidget` with buttons to add/remove rows, load from/save to CSV, for defining PVT table data.

**Worker Threads (`ui/workers/`)**
*   Each inherits `QThread`.
*   Takes necessary data/engine instances in `__init__`.
*   Overrides `run()` to perform the long task.
*   Emits signals (`pyqtSignal`) for progress, completion, errors, and results.

**Utility Modules (`utils/`)**
*   **`project_file_handler.py`:** Contains `save_project_to_tphd(data_dict, filepath)` and `load_project_from_tphd(filepath) -> dict` functions. Handles serialization/deserialization logic including `NumpyEncoder`.
*   **`units_manager.py`:** Manages unit definitions and conversions.
*   **`i18n_manager.py` (Conceptual):** Might contain helper functions for loading translators or managing translatable strings if needed beyond basic Qt `tr()`.
*   **`report_generator.py`:** Logic for creating reports using Jinja2/Markdown/WeasyPrint or ReportLab.

This detailed structure provides a solid framework. Remember to build incrementally, test frequently, and prioritize features based on your PhD research needs. The UI is a significant undertaking, but breaking it into these manageable components will help.