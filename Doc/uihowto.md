Okay, let's refine the UI development instructions, focusing on the core application flow, incorporating your new requirements, and setting aside the plugin architecture for now, while adding AI integration capabilities.

**Latest UI Development Instructions (PyQt6 Focus):**

**I. Core Application Philosophy & Initial View:**

1.  **MainWindow - Overview & Entry Point:**
    *   The application will launch to a clean "Overview" page.
    *   This page should concisely describe the application's purpose (CO₂ EOR evaluation, simulation, and optimization).
    *   It will feature three primary action buttons:
        *   **"Start New Project / Load Data Files":** Transitions the user to the data input and management section.
        *   **"Load Existing Project (.tphd)":** Allows users to open previously saved project files.
        *   **"Quick Start with Defaults":** Enables users to bypass detailed data setup and run the application with pre-configured representative data for demonstration or rapid testing.

**II. Data Input & Management (First Main Section/Tab after Overview):**

1.  **Flexible Data Ingestion:**
    *   Provide distinct UI sections for loading LAS files and Reservoir Model files (Eclipse/GRDECL).
    *   Use file dialogs that allow multi-selection.
    *   Display lists of loaded files with status indicators (e.g., "Parsed OK," "Warning: Missing WELL Name," "Error: Invalid Format").
2.  **User-Driven Processing & Validation:**
    *   A "Process Loaded Files" button to trigger the backend `DataProcessor`.
    *   Display processing logs and a summary of successfully parsed data (number of wells, reservoirs, etc.).
3.  **Interactive Data Review & Override (Post-Processing or for Manual Entry):**
    *   **Well Data:**
        *   Allow selection of individual wells if multiple are loaded.
        *   Display key well information. Implement a "View/Edit Well Logs" dialog.
        *   In this dialog, **crucially prompt for missing essential information (like well names if `MissingWellNameError` occurred during parsing).**
        *   Allow users to specify fill values for problematic curves or mark curves to be ignored during analysis.
    *   **Reservoir Data:**
        *   Display basic parsed reservoir information (filename, grid dimensions).
        *   Implement a "View/Edit Reservoir Properties" dialog.
        *   **OOIP Handling:** Provide options for OOIP:
            *   Display if parsed directly.
            *   Allow manual user input for OOIP (STB).
            *   Implement a dynamic OOIP calculator where users input Average Porosity (Φ), Average Water Saturation (Sw), Net Thickness (h), Area (A), and Oil FVF (Bo). Display the calculated OOIP, and allow the user to accept this value or override it.
            *   Input fields for these parameters should show visible (but distinct, e.g., via placeholder text or a "default:" label) fallback values from the configuration.
    *   **Fluid & PVT Properties:**
        *   Dedicated input area (potentially its own sub-tab or distinct section).
        *   **Attempt to auto-populate from parsed Eclipse data (e.g., PVTO tables) if available.** Users must be able to review this.
        *   **Users must be able to manually define all PVT properties and other fluid characteristics if no data is loaded or if they wish to override.** This includes Bo, μo, Bg, μg, Rs, gas specific gravity, reservoir temperature, and potentially simplified C7+ characteristics or gas composition for MMP calculations.
        *   Use appropriate input widgets (spin boxes for single values, potentially simple table inputs for array-like properties like PVT tables). All fields should allow user input and show configurable fallback values.

**III. Application Configuration (Second Main Section/Tab):**

1.  **Comprehensive Parameter Control:**
    *   Provide dedicated UI sections or expandable groups for each major configuration dataclass (`EORParameters`, `EconomicParameters`, `OperationalParameters`, `ProfileParameters`, `GeneticAlgorithmParamsDefaults`, `RecoveryModelKwargsDefaults`).
2.  **User-Definable Fields:**
    *   Every parameter within these sections must be editable by the user via appropriate input widgets (spin boxes, combo boxes, text inputs).
    *   The UI should clearly display the current active value (loaded from config, project file, or user input).
3.  **Integrated Help ("?" Buttons):**
    *   Place a small "?" button next to each significant parameter or group of parameters.
    *   Clicking this button will display a tooltip, `QMessageBox`, or a small pop-up dialog explaining:
        *   The parameter's definition and purpose.
        *   Its typical range or valid options.
        *   Its impact on the EOR process, economic output, or optimization behavior.
4.  **Fallback Value Visibility:**
    *   For each input field, ensure the underlying default/fallback value (from `config_manager`) is visible to the user, perhaps as placeholder text in the input field or a distinct label (e.g., "Default: 70.0"). This allows users to understand the baseline if they don't provide an override.

**IV. Module-Specific UI Sections (MMP, Optimization, SA, UQ - Subsequent Tabs):**

1.  **MMP & Well Analysis:**
    *   Plotting area for MMP profiles and well logs.
2.  **Optimization Workbench:**
    *   Clear selection for optimization method and target objective.
    *   Dynamic display of parameters relevant to the chosen optimization method.
3.  **Sensitivity Analysis & Uncertainty Quantification:**
    *   **User-Definable Parameters:**
        *   Use a `QTreeView` or an editable `QTableWidget` to allow users to select parameters from various scopes (Reservoir, Fluid, Economic, EOR Operational, Recovery Model internal params) for analysis.
        *   For each selected parameter, provide dynamic input fields for its sensitivity range (min, max, steps, scale) or UQ distribution (type, parameters like mean/std.dev or min/max).
    *   **Fine-Tuning Menus/Dialogs:** For complex multi-input parameters (e.g., defining multiple layers for a `LayeredRecoveryModel` if it's an uncertain variable), use a button to open a dedicated dialog for easier definition.
    *   All settings within these sections must also have integrated "?" help buttons.

**V. AI Integration (New Section/Tab or Integrated into Relevant Workflows):**

1.  **"AI Assistant" or "AI Insights" Section/Tab:**
    *   **API Configuration:** A settings area (perhaps in the global "Settings" dialog or within this AI tab) for users to input their API keys for Gemini, OpenAI, and/or an OpenRouter key, along with base URLs if using self-hosted or alternative endpoints. Store these securely (e.g., using `QSettings` with encryption if possible, or at least clearly warning the user about local storage).
    *   **Model Selection:** `QComboBox` to select the desired AI model (e.g., "Gemini Pro", "GPT-4", "GPT-3.5-turbo", specific models via OpenRouter).
2.  **Potential AI-Assisted Features (Implemented as distinct tools within the AI section):**
    *   **Parameter Justification/Explanation:**
        *   Allow the user to select a parameter from their current configuration (e.g., a specific `EORParameter` value).
        *   A text input for a query like "Explain the impact of this injection rate on CO2 EOR."
        *   The application sends the parameter name, its current value, and the user's query (along with some context about the project, like "CO2 EOR optimization") to the selected AI API.
        *   Display the AI's response in a `QTextBrowser`.
    *   **Sensitivity/UQ Result Interpretation:**
        *   After running SA or UQ, allow the user to select a result (e.g., a tornado chart, a P10/P90 value for NPV).
        *   Prompt: "Provide insights on these sensitivity results for NPV."
        *   Send a summary of the results (e.g., top sensitive parameters, key UQ statistics) to the AI.
        *   Display the AI's interpretation.
    *   **Conceptual Strategy Brainstorming:**
        *   User inputs current reservoir characteristics (e.g., OOIP, depth, temp, MMP) and project goals.
        *   Prompt: "Suggest potential CO2 EOR strategies or parameters to explore for this reservoir."
        *   Display AI suggestions. *Crucially, frame this as brainstorming/idea generation, not definitive advice.*
    *   **Documentation/Report Snippet Generation:**
        *   User selects a section of their results or configuration.
        *   Prompt: "Draft a summary paragraph for a report about these optimized EOR parameters."
        *   Send relevant data to AI and display the drafted text for the user to copy/edit.
    *   **Troubleshooting Assistant (Experimental):**
        *   If an optimization fails or gives unexpected results, the user could describe the issue.
        *   Send logs, configurations, and the problem description to the AI for potential troubleshooting ideas. (Handle data privacy carefully).
3.  **UI for AI Interaction:**
    *   `QTextEdit` for user prompts/queries.
    *   `QTextBrowser` or `QTextEdit` (read-only) to display AI responses.
    *   Buttons: "Send to AI", "Clear Chat/Context".
    *   Consider a simple chat-like interface for ongoing interaction with the AI regarding a specific topic.
4.  **Context Management:** Be mindful of the context window limitations of AI models. Send only relevant information for each query.
5.  **Disclaimer:** Prominently display a disclaimer that AI-generated content is for assistance and brainstorming only, should be critically reviewed, and does not replace expert engineering judgment.

**VI. Cross-Cutting UI Features:**

1.  **Unit System Management:** As previously discussed, allow global selection and ensure all relevant inputs/outputs respect it.
2.  **Advanced Plotting Customization:** For any plot generated, provide options to customize titles, labels, colors, and to export the plot image and its underlying data.
3.  **Report Generation:** Allow users to compile a report including project setup, configurations, key results, plots, and potentially AI-generated summaries.
4.  **Internationalization (i18n - English, Ukrainian):**
    *   Implement Qt's translation mechanism (`QTranslator`, `.ts`/`.qm` files).
    *   Ensure all user-visible strings in the UI are passed through `self.tr()`.
    *   Provide a language selection option in the application settings.

**VII. General Development Workflow:**

1.  **Qt Designer First:** Design individual tabs/dialogs as `.ui` files.
2.  **Compile UI to Python:** Use `pyuic6`.
3.  **Controller Classes:** Create Python classes that inherit from the generated UI class and `QWidget` (or `QDialog`, `QMainWindow`). Implement event handlers (slots) and logic to interact with your backend.
4.  **Threading:** Rigorously use `QThread` for any backend operation that might take more than a fraction of a second to keep the UI responsive. Use signals and slots for communication between threads and the main UI thread.
5.  **Iterative Refinement:** Build and test one feature or tab at a time.

This refined set of instructions provides a more detailed roadmap for developing a comprehensive and user-friendly PyQt6 application for your CO₂ EOR research. The AI integration adds a modern, helpful dimension, but remember to manage user expectations about its capabilities.