# STROBE Statement—Checklist for Observational Studies

**Study Title:** Causal Effects of Electronic Nicotine Delivery Systems (ENDS) Use on Cigarette Smoking: A Double/Debiased Machine Learning Analysis of the PATH Study

---

## Introduction

| Item | Recommendation | Page/Section | Completed |
|------|----------------|--------------|-----------|
| **1. Title and abstract** | (a) Indicate the study's design with a commonly used term in the title or abstract | Title, Abstract | ✓ |
| | (b) Provide in the abstract an informative and balanced summary of what was done and what was found | Abstract | ✓ |

---

## Methods

| Item | Recommendation | Page/Section | Completed |
|------|----------------|--------------|-----------|
| **2. Background/rationale** | Explain the scientific background and rationale for the investigation being reported | Introduction | ✓ |
| **3. Objectives** | State specific objectives, including any prespecified hypotheses | Introduction, Methods | ✓ |
| **4. Study design** | Present key elements of study design early in the paper | Methods | ✓ |
| **5. Setting** | Describe the setting, locations, and relevant dates, including periods of recruitment, exposure, follow-up, and data collection | Methods: Data | ✓ |
| **6. Participants** | (a) Give the eligibility criteria, and the sources and methods of selection of participants | Methods: Sample | ✓ |
| | (b) For matched studies, give matching criteria and number of exposed and unexposed | N/A | N/A |
| **7. Variables** | Clearly define all outcomes, exposures, predictors, potential confounders, and effect modifiers. Give diagnostic criteria, if applicable | Methods: Measures | ✓ |
| **8. Data sources/measurement** | For each variable of interest, give sources of data and details of methods of assessment (measurement). Describe comparability of assessment methods if there is more than one group | Methods: Measures | ✓ |
| **9. Bias** | Describe any efforts to address potential sources of bias | Methods: Identification, Discussion | ✓ |
| **10. Study size** | Explain how the study size was arrived at | Methods: Sample | ✓ |
| **11. Quantitative variables** | Explain how quantitative variables were handled in the analyses. If applicable, describe which groupings were chosen and why | Methods: Analysis | ✓ |
| **12. Statistical methods** | (a) Describe all statistical methods, including those used to control for confounding | Methods: DML | ✓ |
| | (b) Describe any methods used to examine subgroups and interactions | Methods: Heterogeneity | ✓ |
| | (c) Explain how missing data were addressed | Methods: Sample | ✓ |
| | (d) If applicable, explain how loss to follow-up was addressed | Methods: Sample | ✓ |
| | (e) Describe any sensitivity analyses | Methods: Sensitivity | ✓ |

---

## Results

| Item | Recommendation | Page/Section | Completed |
|------|----------------|--------------|-----------|
| **13. Participants** | (a) Report numbers of individuals at each stage of study—e.g., numbers potentially eligible, examined for eligibility, confirmed eligible, included in the study, completing follow-up, and analyzed | Results: Flow | ✓ |
| | (b) Give reasons for non-participation at each stage | Results: Flow | ✓ |
| | (c) Consider use of a flow diagram | Figure 1 | ✓ |
| **14. Descriptive data** | (a) Give characteristics of study participants (e.g., demographic, clinical, social) and information on exposures and potential confounders | Results: Table 1 | ✓ |
| | (b) Indicate number of participants with missing data for each variable of interest | Results: Table 1 | ✓ |
| **15. Outcome data** | Report numbers of outcome events or summary measures over time | Results: Main | ✓ |
| **16. Main results** | (a) Give unadjusted estimates and, if applicable, confounder-adjusted estimates and their precision (e.g., 95% CI). Make clear which confounders were adjusted for and why they were included | Results: Table 2 | ✓ |
| | (b) Report category boundaries when continuous variables were categorized | Results | ✓ |
| | (c) If relevant, consider translating estimates of relative risk into absolute risk for a meaningful time period | Results | ✓ |
| **17. Other analyses** | Report other analyses done—e.g., analyses of subgroups and interactions, and sensitivity analyses | Results: Heterogeneity, Sensitivity | ✓ |

---

## Discussion

| Item | Recommendation | Page/Section | Completed |
|------|----------------|--------------|-----------|
| **18. Key results** | Summarize key results with reference to study objectives | Discussion | ✓ |
| **19. Limitations** | Discuss limitations of the study, taking into account sources of potential bias or imprecision. Discuss both direction and magnitude of any potential bias | Discussion: Limitations | ✓ |
| **20. Interpretation** | Give a cautious overall interpretation of results considering objectives, limitations, multiplicity of analyses, results from similar studies, and other relevant evidence | Discussion | ✓ |
| **21. Generalizability** | Discuss the generalizability (external validity) of the study results | Discussion: Limitations | ✓ |

---

## Other Information

| Item | Recommendation | Page/Section | Completed |
|------|----------------|--------------|-----------|
| **22. Funding** | Give the source of funding and the role of the funders for the present study and, if applicable, for the original study on which the present article is based | Acknowledgments | ✓ |

---

## Additional Considerations for Causal Inference Studies

| Item | Recommendation | Page/Section | Completed |
|------|----------------|--------------|-----------|
| **Causal assumptions** | Clearly state the causal assumptions (unconfoundedness, positivity, SUTVA) | Methods: Identification | ✓ |
| **Identification strategy** | Describe the identification strategy (e.g., selection-on-observables, DAG) | Methods: Identification | ✓ |
| **Causal estimand** | Precisely define the causal estimand (ATE, ATT, etc.) | Methods: Estimands | ✓ |
| **Covariate balance** | Report covariate balance before and after adjustment | Results: Balance | ✓ |
| **Positivity/overlap** | Report and visualize common support/overlap | Results: Diagnostics | ✓ |
| **Sensitivity to confounding** | Conduct sensitivity analysis for unobserved confounding | Results: Sensitivity | ✓ |

---

## Notes

This checklist is adapted from the STROBE Statement (von Elm et al., 2007) with additional items relevant for causal inference studies using observational data. The study employs Double/Debiased Machine Learning (DML) to estimate causal effects while accounting for the complex survey design of the PATH Study through balanced repeated replication (BRR) variance estimation.

**Reference:**
von Elm E, Altman DG, Egger M, Pocock SJ, Gøtzsche PC, Vandenbroucke JP. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. *Lancet*. 2007;370(9596):1453-1457.
