# Data Management Help

<a id="data-management-overview"></a>
## Data Management Overview

This section is for managing all project-related data, including well data, reservoir properties, and PVT information.

<a id="dimensional-consistency"></a>
## Dimensional Consistency for Physics-Based Models

### Overview
The CO2 EOR optimizer enforces dimensional consistency when using physics-based breakthrough models. This ensures that Original Oil In Place (OOIP) is derived from physical reservoir dimensions rather than being an independent variable.

### Physics-Based Models Requiring Dimensional Consistency
The following recovery models require dimensional consistency:
- **Hybrid Model**: Combines miscible and immiscible displacement mechanisms
- **Layered Model**: Accounts for reservoir heterogeneity and layering
- **Buckley-Leverett Model**: Uses fractional flow theory for displacement efficiency
- **Dykstra-Parsons Model**: Calculates vertical sweep efficiency for layered reservoirs

### OOIP Determination Modes

#### Calculate from Parameters (Recommended for Physics-Based Models)
- OOIP is calculated using the volumetric formula:
  ```
  OOIP = 7758 × Area × Thickness × Porosity × (1 - Initial Water Saturation) / Oil FVF
  ```
- This mode is automatically enforced when physics-based models are selected
- Provides physically consistent results for advanced simulations

#### Direct Input (Simple Models Only)
- Allows direct OOIP input for simpler models
- Disabled when physics-based models are selected
- Suitable for screening studies and simple calculations

### Geology Modeling Parameters

#### Enhanced Reservoir Characterization
- **Rock Type**: Classification of reservoir rock (sandstone, carbonate, shale, etc.)
- **Depositional Environment**: Geological setting (fluvial, deltaic, marine, etc.)
- **Structural Complexity**: Rating of reservoir structural features

#### Physical Parameters
- **Area**: Reservoir area in acres or square meters
- **Thickness**: Net pay thickness in feet or meters
- **Porosity**: Average reservoir porosity (0.01 to 0.60)
- **Initial Water Saturation**: Connate water saturation (0.0 to 1.0)
- **Oil FVF**: Initial oil formation volume factor

### Validation and Error Handling

#### Automatic Validation
The system automatically validates dimensional consistency when:
- Physics-based models are selected in optimization
- Reservoir data is processed for simulation
- Optimization parameters are evaluated

#### Common Validation Errors
1. **Missing Parameters**: Required geology parameters not provided
2. **OOIP Inconsistency**: Calculated OOIP differs significantly from provided value
3. **Parameter Range Violation**: Values outside physically realistic ranges

#### Error Resolution
- Check that all required geology parameters are entered
- Ensure OOIP calculation matches provided value within tolerance
- Verify parameter values are within valid ranges
- Consider using simpler models if detailed geology data is unavailable

### Best Practices

#### For Advanced Studies
1. Use physics-based models with complete geology data
2. Ensure dimensional consistency before optimization
3. Validate parameter ranges against field data
4. Use the geology modeling section for comprehensive characterization

#### For Screening Studies
1. Use simpler models (miscible, immiscible, koval)
2. Direct OOIP input is acceptable for screening
3. Focus on key economic and operational parameters

### Benefits of Dimensional Consistency

1. **Physical Realism**: Ensures models are grounded in reservoir physics
2. **Model Integrity**: Prevents unrealistic optimization results
3. **Educational Value**: Teaches proper reservoir engineering principles
4. **Industry Standards**: Aligns with professional reservoir modeling practices

### Troubleshooting

#### OOIP Calculation Issues
- Verify area, thickness, porosity, and saturation values
- Check that Oil FVF is appropriate for reservoir conditions
- Ensure consistent units throughout the calculation

#### Model Selection Guidance
- Use physics-based models for detailed reservoir studies
- Use simpler models for quick screening and sensitivity analysis
- Consider data availability when selecting models

#### Validation Failures
- Review error messages for specific parameter issues
- Check parameter ranges against typical reservoir values
- Consult geological data or analog reservoirs for guidance
