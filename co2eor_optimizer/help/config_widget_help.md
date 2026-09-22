# CO2EOR Optimizer Suite - Complete Help Guide

This guide provides detailed help for all parameters and settings found within the Configuration screen.

---

## **Help: Configuration Overview**

The Configuration screen is the central hub for defining all the parameters that govern the CO2 EOR optimization process. It is organized into several tabs, each focusing on a different aspect of the project.

### Managing Configurations

At the top of the window, you will find several buttons to manage your settings:

*   **Load from File...**: Opens a dialog to load a previously saved configuration from a `.json` file. This is useful for quickly switching between different scenarios or restoring a standard setup.
*   **Save to File...**: Saves the *currently applied* configuration to a `.json` file. **Note:** This will not save any pending changes that you haven't applied yet.
*   **Reset All to Defaults**: Reverts every single parameter on every tab back to the original application defaults. This action cannot be undone.

### Applying and Discarding Changes

As you modify parameters, a yellow banner will appear at the top of the screen indicating you have unsaved changes.

*   **Apply Changes**: Validates and saves your pending changes to the active configuration. The inputs will be checked against validation rules, and any errors will be highlighted. The changes only take effect once they are successfully applied.
*   **Discard Changes**: Reverts all changes you have made since your last "Apply" action, restoring the forms to their previously saved state.

---

## **Help: Economic Parameters**

This section defines the key economic assumptions used in the Net Present Value (NPV) and other financial calculations. These values are primary drivers of project profitability and have a significant impact on the optimizer's decisions.

### <a id="interest-rate"></a>Interest Rate
**A detailed explanation of the annual discount rate used for Net Present Value (NPV) calculations.**

*   **What it is:** The interest rate, or discount rate, represents the time value of money. The core principle is that a dollar today is worth more than a dollar tomorrow because today's dollar can be invested to earn interest. This rate is used to discount future cash flows back to their present value. The formula for the discount factor in a given year `t` is `1 / (1 + rate)^t`.
*   **Why it's important:** It is one of the most sensitive inputs for NPV. A high discount rate heavily penalizes projects with long payback periods, as revenues in the distant future are considered almost worthless. Conversely, a low discount rate makes long-term projects more attractive.
*   **Context and Typical Values:** This rate is often set based on a company's Weighted Average Cost of Capital (WACC), which reflects the blended cost of its debt and equity. For projects with higher perceived risk (like EOR), a "risk premium" might be added, pushing the rate to 10%-20% or even higher. A value of `0.1` represents a 10% annual discount rate.

### <a id="oil-price"></a>Oil Price
**A detailed explanation of the constant price of oil ($/bbl) assumed throughout the project's lifetime.**

*   **What it is:** This is the forecast sale price for each barrel (`bbl`) of crude oil produced by the project. For simplicity, this model assumes a single, constant price for the entire project.
*   **Why it's important:** This parameter is the primary driver of project revenue (`Revenue = Price × Production`). The optimizer's strategy will be heavily influenced by it. High oil prices can justify more aggressive, expensive recovery techniques, while low prices may favor minimizing operational costs.
*   **Context and Interactions:** While oil prices are notoriously volatile in reality, using a constant price is standard for screening-level analysis. The value chosen strongly interacts with the `Min Economic Rate`, as it determines how long a well can profitably produce before being shut-in.

### <a id="co2-purchase-price"></a>CO2 Purchase Price
**A detailed explanation of the cost to purchase CO2 ($/tonne) from an external source.**

*   **What it is:** This parameter defines the direct cost of acquiring one metric tonne of CO2 before it is injected. It does not include the cost of transportation or injection.
*   **Why it's important:** This is a major operational expenditure (OPEX) and a primary cost driver for any CO2 EOR project. The optimizer must balance the cost of injecting more CO2 (to potentially recover more oil) against the price of that CO2.
*   **Context and Interactions:** The price can vary dramatically depending on the source (e.g., low-cost natural CO2 domes vs. high-cost industrial direct air capture). This parameter has a strong inverse relationship with `CO2 Recycling Efficiency`; if the purchase price is very high, investing in highly efficient recycling facilities becomes much more economically attractive.

### <a id="co2-transport-cost"></a>CO2 Transport Cost
**A detailed explanation of the cost to transport purchased CO2 ($/tonne) to the project site.**

*   **What it is:** This represents the logistical cost (per tonne) associated with moving the purchased CO2 from its source to the project's injection facilities.
*   **Why it's important:** This cost is added to the `CO2 Purchase Price` to get the total "landed cost" of new CO2. In some cases, especially for remote fields far from a CO2 source, transport costs can be as significant as the purchase price itself.
*   **Context and Typical Values:** Costs are lowest for established pipeline infrastructure. They increase significantly for transport by truck, rail, or ship. This parameter forces the model to consider the project's geographical context.

---

## **Help: EOR Parameters**

This section defines the core technical parameters related to the Enhanced Oil Recovery process itself, dictating the interaction between the injected fluids and the reservoir.

### <a id="min-miscibility-pressure"></a>Minimum Miscibility Pressure (MMP)
**A detailed explanation of the minimum reservoir pressure required for CO2 to become miscible with the oil.**

*   **What it is:** Miscibility is the point at which CO2 and oil mix in all proportions to form a single fluid phase, much like alcohol and water. This eliminates the interfacial tension between the fluids, which dramatically reduces the residual oil saturation and improves oil displacement efficiency. The MMP (measured in `psi`) is the pressure threshold required to achieve this state at a given reservoir temperature and oil composition.
*   **Why it's important:** Operating the flood above the MMP is a primary goal for most CO2 EOR projects. The optimizer may prioritize injection strategies that maintain reservoir pressure above this critical value to maximize oil recovery.
*   **Context:** MMP is not a fixed number; it is determined through laboratory experiments (e.g., slim-tube tests) or equations of state modeling. It is highly sensitive to the presence of lighter hydrocarbons in the oil and impurities in the injected CO2.

### <a id="wag-co2-slug-size"></a>WAG CO2 Slug Size (PV)
**A detailed explanation of the size of the CO2 injection 'slug' as a fraction of pore volume in a WAG scheme.**

*   **What it is:** In a Water-Alternating-Gas (WAG) scheme, "slugs" of CO2 and water are injected in turns. This parameter defines the volume of the CO2 slug as a fraction of the total reservoir Pore Volume (PV). Pore volume is the total empty space within the rock that can hold fluid.
*   **Why it's important:** The slug size is a key optimization variable. Small slugs may not form a continuous, effective miscible bank. Large slugs use more expensive CO2 and may lead to earlier CO2 breakthrough.
*   **Example:** If a reservoir has a total pore volume of 10 million barrels and this parameter is set to `0.05`, each CO2 injection slug will have a volume of 500,000 barrels.

### <a id="first-co2-injection-phase"></a>First CO2 Injection Phase Duration (Years)
**A detailed explanation of the duration in years of the initial CO2 injection phase.**

*   **What it is:** This sets the length of the very first injection period. It is common practice to begin a project with a large, continuous slug of CO2.
*   **Why it's important:** The purpose of this initial phase is to quickly pressurize the near-wellbore region above the MMP and to establish a robust "miscible bank" that will form the leading edge of the displacement front. The duration is a trade-off between ensuring miscibility and the high upfront cost of the CO2.

### <a id="subsequent-co2-injection-phase"></a>Subsequent CO2 Injection Phase Duration (Years)
**A detailed explanation of the duration of subsequent CO2 injection cycles in a WAG scheme.**

*   **What it is:** After the initial phase, this parameter defines how long each follow-up CO2 injection period lasts within a WAG cycle.
*   **Why it's important:** This, along with the water phase duration, defines the WAG ratio and cycle time, which are critical for controlling fluid mobility and sweep efficiency. Shorter, more frequent cycles can provide better mobility control but may be more complex operationally.

### <a id="subsequent-water-injection-phase"></a>Subsequent Water Injection Phase Duration (Years)
**A detailed explanation of the duration of subsequent water injection cycles in a WAG scheme.**

*   **What it is:** This defines the duration of the water injection period that follows a CO2 injection period in a WAG cycle.
*   **Why it's important:** The primary purpose of the water slug is to improve sweep efficiency. Water is more viscous than CO2 and has a more favorable mobility ratio with oil, so it acts to "push" the CO2 bank, preventing severe fingering and forcing the CO2 into parts of the reservoir it would otherwise bypass. The ratio of the water duration to the CO2 duration (the WAG ratio) is a critical optimization parameter.

---

## **Help: Operational Parameters**

This tab controls the high-level operational constraints and the overall strategy of the optimization process.

### General Parameters

#### <a id="project-lifetime"></a>Project Lifetime (Years)
**A detailed explanation of the total duration of the EOR project in years.**

*   **What it is:** This sets the overall time horizon for all simulations and financial calculations.
*   **Why it's important:** It defines the "t" variable in NPV calculations and sets the boundary for production profile modeling. A longer lifetime allows for slower, potentially more efficient recovery strategies but discounts later-year revenues more heavily. A shorter lifetime forces more aggressive, front-loaded production schedules. It is a fundamental constraint for the entire optimization.

#### <a id="max-injection-rate"></a>Max Injection Rate Per Well (bpd)
**A detailed explanation of the maximum physical injection rate (bpd) for a single injection well.**

*   **What it is:** This represents the maximum volumetric rate (in barrels per day, `bpd`) at which fluid can be injected into a single well.
*   **Why it's important:** This is a critical real-world constraint. The rate is limited by the reservoir's "injectivity" (a function of rock permeability and fluid viscosity), the pressure rating of the surface pumps and pipes, and the fracture pressure of the reservoir rock (injecting at too high a pressure will crack the rock, which can be undesirable). The total field injection rate is this value multiplied by the number of active injectors.

#### <a id="injection-scheme"></a>Injection Scheme
**A detailed explanation of the overall strategy for injection: continuous CO2 or Water-Alternating-Gas (WAG).**

*   **What it is:** This defines the fundamental fluid injection strategy.
*   **Why it's important:** The choice has massive implications for cost, complexity, and recovery.
    *   `continuous`: A simpler strategy where only CO2 is injected. It can result in faster oil recovery initially but is often prone to poor sweep efficiency due to viscous fingering and gravity override, leading to early CO2 breakthrough and bypassing large volumes of oil.
    *   `wag`: A more complex strategy that alternates between injecting CO2 and water. The water injection phase helps to control the mobility of the less-viscous CO2, resulting in a more stable displacement front and better volumetric sweep of the reservoir. This often leads to higher ultimate recovery, though it may be slower and requires dual-fluid infrastructure.

### Optimizer Target Seeking

This group of settings allows you to guide the optimizer towards a specific goal, rather than just maximizing a value.

#### <a id="enable-target-seeking"></a>Enable Target Seeking
**A detailed explanation of enabling the optimizer to aim for a specific target value for a chosen objective.**

*   **What it is:** This checkbox fundamentally changes the optimizer's objective function. Instead of simple maximization (e.g., "get the highest NPV possible"), it switches to a target-seeking mode.
*   **Why it's important:** This is useful for "what-if" scenarios or situations with external constraints. For example, if a company has a contractual obligation to sequester a specific amount of CO2, you can use this mode to find the most profitable way to operate *while meeting that specific storage target*.

#### <a id="target-objective"></a>Target Objective
**A detailed explanation of the performance metric (e.g., NPV, Recovery Factor) to target.**

*   **What it is:** A dropdown menu to select the performance metric that the optimizer should target.
*   **Why it's important:** It allows the user to specify the goalpost for the target-seeking operation. You can set a target for a financial metric (NPV), a technical metric (Recovery Factor), or an environmental metric (CO2 Utilization).

#### <a id="target-value"></a>Target Value
**A detailed explanation of the specific numerical value the optimizer will attempt to achieve.**

*   **What it is:** The numerical goal for the selected `Target Objective`.
*   **Why it's important:** This sets the precise value the optimizer will try to hit. The objective function will reward solutions that are closer to this value and penalize those that are further away.

#### <a id="target-seeking-sharpness"></a>Target Seeking Sharpness
**A detailed explanation of the factor controlling how aggressively the optimizer seeks a specific target value.**

*   **What it is:** An advanced parameter that tunes the shape of the penalty function around the target value.
*   **Why it's important:** It controls the trade-off between hitting the target precisely and achieving good performance on other, secondary metrics.
    *   **High Sharpness:** Creates a very narrow, steep penalty function. The optimizer will be heavily penalized for even small deviations from the target, forcing it to prioritize hitting the target above all else.
    *   **Low Sharpness:** Creates a broader, gentler penalty function. The optimizer has more flexibility to find solutions that are "close enough" to the target but might be significantly better in terms of, for example, profitability.

---

## **Help: Production Profile Parameters**

This section controls the analytical "type curve" models used to forecast oil production and CO2 handling. These simplified models are used in place of full, time-consuming reservoir simulations.

### <a id="oil-profile-type"></a>Oil Profile Type
**A detailed explanation of the analytical model used to describe the oil production rate over time.**

*   **What it is:** This is the primary choice that determines the mathematical shape of the production curve over the project's life.
*   **Why it's important:** The choice of model dictates the entire production forecast. "Plateau" models are common for new projects, representing an initial period of facility-constrained production followed by a natural decline. The different decline types (Exponential, Hyperbolic) represent different physical reservoir drive mechanisms.
*   **Context:** These are empirical models, most famously described by Arps. They provide a fast and effective way to approximate the results of complex numerical simulations.

### <a id="injection-profile-type"></a>Injection Profile Type
**A detailed explanation of the model used to describe the CO2 injection rate over time.**

*   **What it is:** Defines how the injection rate is modeled. `Constant During Each Phase` is a standard simplification.
*   **Why it's important:** It assumes that during a CO2 injection cycle, the rate is held at a steady, optimized value. In reality, rates might vary, but this is a reasonable assumption for a long-term strategic model.

### <a id="oil-annual-fraction"></a>Oil Annual Fraction of Total
**A detailed explanation of defining a custom production profile via a list of annual production fractions.**

*   **What it is:** An advanced option that allows the user to bypass the analytical models and define the production profile directly. The user provides a comma-separated list of values, where each value is the fraction of the total ultimate recovery produced in that year.
*   **Why it's important:** This is useful for history matching or when a production profile from an external, more sophisticated reservoir simulator is available.
*   **Constraints:** The number of entries in the list **must** equal the `Project Lifetime`, and the sum of all fractions **must** equal 1.0.

### <a id="plateau-duration"></a>Plateau Duration (Fraction of Life)
**A detailed explanation of the duration of the peak (plateau) production period as a fraction of project life.**

*   **What it is:** The initial period of a project's life where production is held at a constant, maximum rate. This rate is typically constrained by the capacity of surface facilities (separators, pumps, pipelines), not by the reservoir's ability to produce.
*   **Why it's important:** A longer plateau period means more oil is recovered earlier, which is highly favorable for the project's NPV. The optimizer will often seek to maximize the plateau duration within other constraints.

### <a id="initial-decline-rate"></a>Initial Decline Rate (Annual Fraction)
**A detailed explanation of the initial rate of production decline after the plateau period ends.**

*   **What it is:** This marks the beginning of the "decline phase," where production is no longer constrained by facilities but by the reservoir itself (e.g., falling pressure). This parameter is the *nominal* decline rate at the instant the plateau ends.
*   **Why it's important:** It sets the steepness of the initial production fall-off. A high decline rate means production drops quickly, negatively impacting revenues. It is a key characteristic of the reservoir's energy and fluid properties.

### <a id="hyperbolic-b-factor"></a>Hyperbolic B-Factor
**A detailed explanation of the 'b' exponent for the hyperbolic decline curve model.**

*   **What it is:** This factor, used only in hyperbolic decline, describes how the decline rate itself changes over time. It must be between 0 and 1.
*   **Why it's important:** It provides a more realistic decline model than simple exponential decline.
    *   `b = 0` (Exponential Decline): The decline rate is constant (e.g., production drops by 10% *of the current rate* every year). This is a pessimistic case.
    *   `b = 1` (Harmonic Decline): The decline rate decreases very slowly. This is a very optimistic case.
    *   `0 < b < 1` (Hyperbolic Decline): The most common case. The decline rate lessens over time as the reservoir's energy depletes more slowly.

### <a id="min-economic-rate"></a>Min Economic Rate (Fraction of Peak)
**A detailed explanation of the production rate at which the project becomes uneconomical.**

*   **What it is:** This defines the "economic limit." It is the production rate below which the revenue from selling the oil is less than the operational costs of extracting it. Production ceases once this rate is reached. It is expressed as a fraction of the peak (plateau) rate.
*   **Why it's important:** It determines the effective producing life of the project. A high economic limit (e.g., due to high operational costs) will cut the project's life short, leaving recoverable oil in the ground. It interacts strongly with `Oil Price`.

### <a id="co2-breakthrough"></a>CO2 Breakthrough Year (Fraction)
**A detailed explanation of the time at which injected CO2 begins to be produced.**

*   **What it is:** The point in time, expressed as a fraction of the project life, when the injected CO2 front reaches a production well. Before this time, only oil and formation water are produced. After this time, produced fluids contain CO2.
*   **Why it's important:** Early breakthrough is generally undesirable. It indicates poor sweep efficiency (the CO2 has found a "shortcut" to the producer) and marks the beginning of CO2 production, which must be handled, separated, and either vented or recycled, incurring costs.

### <a id="co2-production-ratio"></a>CO2 Production Ratio after Breakthrough
**A detailed explanation of the ratio of produced CO2 to injected CO2 after breakthrough occurs.**

*   **What it is:** After breakthrough, this parameter defines the volume of CO2 produced for every volume of CO2 injected. It models the "Gas-Oil Ratio" (GOR) of the produced fluid stream.
*   **Why it's important:** A low ratio is desirable, indicating that the injected CO2 is spending its time displacing oil rather than cycling directly from injector to producer. A high ratio indicates poor conformance and high recycling costs. For simplicity, this model assumes the ratio is constant after breakthrough.

### <a id="co2-recycling-efficiency"></a>CO2 Recycling Efficiency (Fraction)
**A detailed explanation of the fraction of produced CO2 that can be successfully captured and re-injected.**

*   **What it is:** This represents the overall efficiency of the surface gas processing facilities. Produced fluids are separated, and the CO2 is stripped from the gas stream, re-compressed, and sent back to the injection wells.
*   **Why it's important:** This process is never 100% efficient, and some CO2 is always lost. High recycling efficiency is crucial for minimizing costs, as every tonne of recycled CO2 is one less tonne that needs to be purchased. The required investment in facilities is a trade-off against the `CO2 Purchase Price`.

### <a id="warn-if-defaults-used"></a>Warn if Defaults Used
**A detailed explanation of logging a warning if the default production profile is used without user modification.**

*   **What it is:** A simple safety-check checkbox.
*   **Why it's important:** If enabled, the application will remind the user via a log message that the simulation is running on a generic, non-specific production profile. This is to prevent users from accidentally making critical decisions based on default data that may not be representative of their specific reservoir.

---

## **Help: Genetic Algorithm Parameters**

The Genetic Algorithm (GA) is a search heuristic inspired by natural selection. It evolves a "population" of candidate solutions over many "generations" to find an optimal result. It is robust and good for exploring complex, poorly understood solution spaces.

### <a id="population-size"></a>Population Size
**A detailed explanation of the number of individuals (solutions) in each generation.**

*   **What it is:** The number of distinct candidate solutions that the algorithm maintains at any one time.
*   **Why it's important:** It controls the diversity of the "gene pool."
    *   **Larger Population:** Explores the search space more thoroughly and is less likely to get stuck in a poor local optimum. However, it requires more simulation runs per generation, making the optimization much slower.
    *   **Smaller Population:** Converges faster but runs a higher risk of converging prematurely to a suboptimal solution.

### <a id="crossover-rate"></a>Crossover Rate
**A detailed explanation of the probability (0 to 1) that two individuals will "mate" to create offspring.**

*   **What it is:** The probability that two parent solutions, selected based on their "fitness" (e.g., their NPV), will be combined to create one or more new child solutions for the next generation.
*   **Why it's important:** Crossover is the primary mechanism for convergence. It promotes the combination of "good" features from different solutions. A typical value is high, e.g., `0.8` (80%).

### <a id="mutation-rate"></a>Mutation Rate
**A detailed explanation of the probability (0 to 1) of random changes occurring in an individual's genes.**

*   **What it is:** The probability that a random parameter within a single solution will be changed to a new random value.
*   **Why it's important:** Mutation is essential for maintaining genetic diversity and preventing stagnation. It allows the algorithm to escape local optima and explore entirely new regions of the search space. The rate is typically kept very low, e.g., `0.01` to `0.1` (1-10%), to avoid disrupting otherwise good solutions.

### <a id="number-of-generations"></a>Number of Generations
**A detailed explanation of the total number of generations the algorithm will run before stopping.**

*   **What it is:** The primary stopping criterion for the optimizer. The algorithm will evolve the population for this many generations.
*   **Why it's important:** It directly controls the total runtime of the optimization. More generations allow the algorithm more time to converge on a good solution, but the computational cost is linear. The optimal number is problem-dependent and often found through trial and error.

---

## **Help: Bayesian Optimization Parameters**

Bayesian Optimization is a highly efficient sequential optimization technique, ideal for problems where the objective function (i.e., the simulation) is very expensive to evaluate. It builds a statistical model of the problem to intelligently choose the next best point to test.

### <a id="bo-iterations"></a>BO Iterations
**A detailed explanation of the total number of iterations for the Bayesian Optimization process.**

*   **What it is:** The total number of full simulation runs the algorithm will perform.
*   **Why it's important:** Unlike a GA, each iteration is not a "generation" but a single, carefully chosen simulation. Because the algorithm learns from each run and makes an informed decision about the next point to try, it can often find excellent solutions with a fraction of the simulation runs required by a GA. This makes it ideal for computationally expensive problems.

### <a id="acquisition-function"></a>Acquisition Function
**A detailed explanation of the strategy used to select the next point to evaluate.**

*   **What it is:** This is the mathematical function that guides the search by balancing exploration and exploitation.
*   **Why it's important:** It is the "brain" of the Bayesian optimizer.
    *   **Exploitation:** Testing points where the underlying statistical model predicts a high objective value. This is like drilling a new well right next to your best-producing well.
    *   **Exploration:** Testing points where the model is most uncertain. This reduces overall uncertainty and prevents the model from ignoring a potentially excellent but unexplored region of the search space. This is like drilling a "wildcat" well in a new area.
*   **Common Choices:** `ei` (Expected Improvement) is a popular, well-balanced function and a good default choice.

---

## **Help: Recovery Models Parameters**

This section allows you to tune the parameters for the underlying analytical models that calculate oil recovery based on physical principles.

### Koval Model
The Koval model is a classic method used to predict recovery in miscible floods, specifically accounting for the negative impact of viscous fingering in a simplified way.

#### <a id="koval-vdp-coefficient"></a>V_dp Coefficient
**A detailed explanation of the Koval model parameter representing the viscous-to-gravity force ratio.**

*   **What it is:** A dimensionless parameter in the Koval theory that quantifies the interaction between viscous forces (which cause fingering) and gravity forces (which can help stabilize the flood front).
*   **Why it's important:** It helps the simplified model account for the complex interplay of forces that determine sweep efficiency in a real miscible flood.

#### <a id="koval-mobility-ratio"></a>Mobility Ratio
**A detailed explanation of the ratio of the mobility of the displacing fluid (CO2) to the displaced fluid (oil).**

*   **What it is:** Mobility is defined as permeability divided by viscosity (`k/μ`). The mobility ratio `M` is the mobility of the displacing fluid (CO2) divided by the mobility of the displaced fluid (oil).
*   **Why it's important:** It is a fundamental indicator of flood stability.
    *   `M <= 1`: Stable displacement. The front moves uniformly.
    *   `M > 1`: Unstable displacement. The less-viscous CO2 will "finger" through the more-viscous oil, leading to poor sweep efficiency and early breakthrough. CO2 EOR floods typically have highly unfavorable mobility ratios. The Koval model uses `M` to predict the severity of this effect.

### Miscible Model
Parameters for a general-purpose miscible displacement model, focusing on gravity effects.

#### <a id="miscible-kv-factor"></a>Kv Factor
**A detailed explanation of the vertical to horizontal permeability ratio (Kv/Kh).**

*   **What it is:** The ratio of the rock's permeability in the vertical direction (`Kv`) to its permeability in the horizontal direction (`Kh`).
*   **Why it's important:** This factor is crucial for modeling "gravity override." Because CO2 is much less dense than oil, it will tend to rise to the top of the reservoir and flow along the caprock, bypassing oil in the lower sections. A high `Kv` allows this vertical segregation to happen more easily, worsening the effect. A `Kv/Kh` of `0.1` is a common rule of thumb for many reservoir types.

#### <a id="miscible-gravity-factor"></a>Gravity Factor
**A detailed explanation of a factor accounting for the general effects of gravity on fluid segregation.**

*   **What it is:** A more general, empirical term that models the overall impact of gravity on the flood's vertical sweep efficiency.
*   **Why it's important:** It provides a lever to tune the model's gravity segregation behavior, especially when detailed Kv/Kh data is not available.

### Immiscible Model
Parameters for an immiscible displacement model, where CO2 and oil do not mix to a single phase.

#### <a id="immiscible-sor"></a>Residual Oil Saturation (Sor)
**A detailed explanation of Residual Oil Saturation, the fraction of oil left behind in swept zones.**

*   **What it is:** `Sor` represents the fraction of oil that is physically unrecoverable in the parts of the reservoir that have been contacted by the displacing fluid. It is oil that is trapped in tiny pore throats by capillary forces or stuck to the surface of the rock grains.
*   **Why it's important:** It defines the "microscopic" displacement efficiency. The total recovery is a product of this microscopic efficiency and the "macroscopic" or volumetric sweep efficiency (the fraction of the reservoir contacted). Lowering `Sor` is a primary goal of any EOR process. Miscible floods can theoretically achieve an `Sor` of zero.

#### <a id="immiscible-krw-max"></a>Max Relative Permeability to Water (Krw_max)
**A detailed explanation of the maximum relative permeability to water.**

*   **What it is:** Permeability is the rock's ability to transmit fluid. *Relative* permeability is a dimensionless correction factor (from 0 to 1) that describes how well one fluid can flow when other fluids are also present in the pores. `Krw_max` is the endpoint of the water relative permeability curve, representing the permeability to water when the oil saturation has been reduced to its residual level (`Sor`).
*   **Why it's important:** This parameter is a critical input for multiphase flow calculations using Darcy's Law. It governs the injectivity of water and the pressure response during a water or WAG flood, directly impacting sweep efficiency and operational pressures.