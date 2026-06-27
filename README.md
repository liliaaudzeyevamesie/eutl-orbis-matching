# eutl-orbis-matching
Tiered matching procedure linking EU ETS participation data from the EUTL to firm-level financial characteristics from Orbis, following Chan et al. (2013). Replication code for my working paper: "Carbon Regulation and Firm Investment: Evidence from EU ETS Phase IV".

# EU ETS - ORBIS Tiered Matching Script

This repository contains the matching code used in:

**[Your Name]. "Carbon Regulation and Firm Investment: Evidence from EU ETS Phase IV." Working Paper, Durham University, 2025. [SSRN link]**

## Overview

This script matches EU ETS installation-level compliance data to firm-level financial data from the Orbis database using a tiered matching procedure, following the methodology of Chan et al. (2013). It produces a firm-year panel dataset for use in difference-in-differences analysis.

## Matching Procedure

Matching is conducted in two tiers:

**Tier 1 (Exact matching):**
- Tier 1A: Company registration number exact match
- Tier 1B: Company name + postal code
- Tier 1C: Company name + city

**Tier 2 (Fuzzy matching):**
- Fuzzy name matching by country for remaining unmatched installations
- Similarity threshold of 0.85 (chosen based on manual validation)

## Required Input Files

- `orbis_data.csv` — consolidated Orbis financial data
- `operators_daily.csv` — EU ETS operators data
- `compliance_2021.csv`, `compliance_2022.csv`, `compliance_2023.csv`, `compliance_2024.csv` — EU ETS compliance data by year

## Output Files

- `matched_euets_orbis_final.csv` — firm-year panel dataset (treated and control firms)
- `matching_details_tiered.csv` — detailed match results by tier

## Requirements

Install dependencies with: `pip install pandas numpy rapidfuzz`

Note: `rapidfuzz` is optional but recommended for significantly faster fuzzy matching. The script will fall back to `difflib` if not installed.

## Usage

Run with: `python matching_script.py`

Ensure all input files are in the same directory as the script before running.

## Notes

- Sample covers nine European countries: AT, BE, CZ, DE, ES, FR, IT, NL, PL
- Compliance data covers EU ETS Phase IV years 2021-2024
- Financial panel spans 2013-2024
- Column names in Steps 9-10 reflect this study's Orbis export format and may need adjustment for different exports
- `Tobins_Q` is included in the output for potential future use but is not used in the analysis
- Runtime depends on sample size; Tier 2 fuzzy matching is the most computationally intensive step

## Reference

Chan, H.S., Li, S. and Zhang, F. (2013). Firm competitiveness and the European Union emissions trading scheme. Energy Policy, 63, pp.1056-1064.
