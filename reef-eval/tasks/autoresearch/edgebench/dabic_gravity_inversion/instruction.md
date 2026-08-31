Task objective: Port the D-ABIC method proposed by Song Han et al. in the 2025 Geophysics paper doi 10.1190/geo-2025-0233 to 3D gravity inversion, and validate it on the measured Vinton salt dome dataset. Read `starter/README.md` for detailed instructions.

The inversion must be completed separately under both L0 and L1 sparse regularization norms. For each norm, run a three-way comparison among D-ABIC, Cooling, and L-curve. Xu et al.'s 2025 Geophysical Prospecting paper doi 10.1111/1365-2478.70016 performed gravity inversion on the same Vinton dataset using an HMC method; use it as the comparison benchmark.

Four stages:
1. Algorithm implementation: read `docs/geo20250233.pdf` and implement a D-ABIC beta-adaptive directive. The suggested name is `class DABIC_Beta_Estimator(directives.InversionDirective)`, placed in a separate module `outputs/dabic_directive.py`. Decide for yourself whether to use the model-space or data-space form, how to compute determinants, and in which space to optimize beta. The directive must work under both the L0 and L1 norms.
2. Synthetic validation: `starter/starter.py` provides the synthetic density model Model 3, survey points, noisy observed data, and forward simulation object, but does not include an inversion framework. Build the SimPEG inversion workflow yourself, and implement both Cooling and L-curve scan baselines. Run the three-way comparison for both L0 and L1.
3. Field-data application: `data/saltdome_s7_100.grd` is the measured Vinton salt dome gravity-anomaly data, in Surfer 7 binary format. It can be parsed by `starter/explore_xu_data.py`. Refer to `docs/Xu2025_HMC.pdf` for the inversion mesh, depth range, density bounds, and related settings.
4. Result comparison: compare quantitatively and qualitatively with the Xu 2025 HMC results, and write `outputs/report.md`, 800 to 1500 words/characters in length.

Deliver to `outputs/`: dabic_directive.py, run_synthetic.py, run_vinton.py, results.json, report.md.

All dependencies are preinstalled; you do not need to and cannot install additional packages. The container has no network access and cannot access external resources.

Constraints: beta must not be hard-coded as a constant; do not import any third-party ABIC package; do not use SimPEG's built-in high-level automatic beta directives; BetaEstimate_ByEig may only be used to set the initial beta_0 value; both L0 and L1 inversions must be fully completed; do not expand the work to other regularization strategies such as GCV or HMC; do not modify starter/explore_xu_data.py or starter/starter.py.
