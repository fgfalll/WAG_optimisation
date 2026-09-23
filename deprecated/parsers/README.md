# Deprecated Parsers Archive

This directory archives legacy file format parsers and validation routines that have been retired from the active production runtime.

## Archived Components

| File | Original Location | Reason for Deprecation |
|---|---|---|
| `base_parser.py` | `parsers/base_parser.py` | Abstract OOP parser class. Zero active callers; superseded by direct utility functions (`utils/las_parser.py`). |
| `validation.py` | `parsers/validation.py` | 3D grid checks for legacy ECLIPSE block decks. Superseded by `core/data_integration_engine.py:DataValidator`. |
| `eclipse_parser.py` | `co2eor_optimizer/parsers/eclipse_parser.py` | Full ECLIPSE deck parser. The application consolidated around the single physics-informed surrogate simulator (`core/engine_surrogate/`), deprecating 3D grid deck ingestion. |

## Active Well Log Ingestion

Petrophysical well log parsing is actively maintained and serviced by:
* [`utils/las_parser.py`](file:///d:/rep/4.6/co2eor_optimizer/utils/las_parser.py) (uses `lasio`).
