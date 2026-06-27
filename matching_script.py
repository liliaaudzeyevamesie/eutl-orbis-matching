"""
EU ETS - ORBIS Tiered Matching Script
=========================================================
Optimized matching using vectorized pandas operations.

Tier 1: Match via operators_daily.csv using:
  - Company registration numbers (exact match)
  - Company name + postal code (strong match)
  - Company name + city (good match)

Tier 2: Fuzzy name matching by country for remaining installations

Requires:
- orbis_data.csv (consolidated ORBIS data with Investments and Age)
- operators_daily.csv (EU ETS operators data)
- compliance_2021.csv, compliance_2022.csv, compliance_2023.csv, compliance_2024.csv
"""

import pandas as pd
import numpy as np
import sys
import time
import re

# Try rapidfuzz
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
    print("[OK] Using rapidfuzz for fast fuzzy matching")
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False
    print("[WARNING] Using difflib (install rapidfuzz for 10x speed: pip install rapidfuzz)")

print("=" * 80)
print("EU ETS - ORBIS TIERED MATCHING (WITH INVESTMENTS & AGE)")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Threshold chosen based on manual validation of match quality for this sample.
# Adjust based on your own validation if applying to different data.
SIMILARITY_THRESHOLD = 0.85
# Countries selected based on Orbis data availability and EU ETS coverage.
# Modify to reflect your own sample selection.
KEEP_COUNTRIES = ['AT', 'BE', 'CZ', 'DE', 'ES', 'FR', 'IT', 'NL', 'PL']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_name(name):
    """Standardize company names."""
    if pd.isna(name):
        return ""
    
    name = str(name).upper()
    legal_forms = [' LTD', ' LIMITED', ' GMBH', ' GESELLSCHAFT', ' AG', ' SA', ' SPA', 
                   ' BV', ' NV', ' AS', ' OY', ' AB', ' SRL', ' SRO', ' SPOL',
                   ' INC', ' CORP', ' CORPORATION', ' PLC', ' LLC', ' LLP',
                   ' CO', ' COMPANY', '& CO', ' KG', ' OHG', ' SARL', ' SAS', ' SE']
    
    for form in legal_forms:
        name = name.replace(form, '')
    
    name = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in name)
    return ' '.join(name.split()).strip()

def normalize_reg_number(reg_num):
    """Normalize registration numbers."""
    if pd.isna(reg_num):
        return ""
    
    reg_num = str(reg_num).upper().strip()
    reg_num = re.sub(r'\([^)]*\)', '', reg_num)  # Remove parentheses
    reg_num = reg_num.replace(' ', '').replace('-', '').replace('.', '')
    
    # Handle scientific notation
    if 'E+' in reg_num or 'E-' in reg_num:
        try:
            reg_num = str(int(float(reg_num)))
        except:
            pass
    
    return reg_num.strip()

def clean_postal_code(code):
    if pd.isna(code):
        return ""
    return str(code).replace(' ', '').replace('-', '').upper().strip()

def clean_city(city):
    if pd.isna(city):
        return ""
    return str(city).upper().strip()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[STEP 1] Loading data...")

# Load compliance
# Compliance data covers Phase IV years 2021-2024. Adjust for different phases.
comp_data = []
for year in [2021, 2022, 2023, 2024]:
    df = pd.read_csv(f'compliance_{year}.csv', encoding='utf-8')
    df['YEAR'] = year
    comp_data.append(df)
    print(f"  [OK] {year}: {len(df):,} installations")

compliance = pd.concat(comp_data, ignore_index=True)
compliance['COUNTRY'] = compliance['REGISTRY_CODE'].str[:2]
compliance = compliance[compliance['COUNTRY'].isin(KEEP_COUNTRIES)]
print(f"  Total compliance (filtered): {len(compliance):,}")

# Load operators
operators = pd.read_csv('operators_daily.csv', encoding='utf-8')
print(f"  [OK] Operators: {len(operators):,}")

# Load ORBIS
orbis = pd.read_csv('orbis_data.csv', encoding='utf-8')
print(f"  [OK] ORBIS: {len(orbis):,} companies")

# ============================================================================
# STEP 2: LINK COMPLIANCE TO OPERATORS
# ============================================================================

print("\n[STEP 2] Linking compliance to operators...")

comp_ops = compliance.merge(
    operators[['INSTALLATION_IDENTIFIER', 'ACCOUNT_HOLDER_NAME', 
               'ACCOUNT_HOLDER_COMPANY_REGISTRATION_NUMBER',
               'ACCOUNT_HOLDER_POSTAL_CODE', 'ACCOUNT_HOLDER_CITY',
               'ACCOUNT_HOLDER_COUNTRY_CODE']],
    on='INSTALLATION_IDENTIFIER',
    how='left'
)

linked = comp_ops[comp_ops['ACCOUNT_HOLDER_NAME'].notna()].copy()
print(f"  [OK] Installations with operator info: {len(linked):,}")

# ============================================================================
# STEP 3: CLEAN DATA (VECTORIZED)
# ============================================================================

print("\n[STEP 3] Cleaning data (vectorized)...")

# Clean operators data
linked['ACC_NAME_CLEAN'] = linked['ACCOUNT_HOLDER_NAME'].apply(clean_name)
linked['ACC_POSTAL_CLEAN'] = linked['ACCOUNT_HOLDER_POSTAL_CODE'].apply(clean_postal_code)
linked['ACC_CITY_CLEAN'] = linked['ACCOUNT_HOLDER_CITY'].apply(clean_city)
linked['ACC_REG_CLEAN'] = linked['ACCOUNT_HOLDER_COMPANY_REGISTRATION_NUMBER'].apply(normalize_reg_number)

# Clean ORBIS data
orbis['NAME_CLEAN'] = orbis['Company name Latin alphabet'].apply(clean_name)
orbis['POSTAL_CLEAN'] = orbis['Standardized postal code'].apply(clean_postal_code)
orbis['CITY_CLEAN'] = orbis['Standardized city'].apply(clean_city)

# Normalize ALL national IDs in ORBIS (handle semicolon-separated lists)
print("  Normalizing national IDs...")
orbis['NATIONAL_ID_NORMALIZED'] = orbis['National ID'].apply(
    lambda x: '; '.join([normalize_reg_number(id.strip()) for id in str(x).split(';')]) if pd.notna(x) else ""
)

print(f"  [OK] Cleaned all data")

# ============================================================================
# STEP 4: TIER 1A - REGISTRATION NUMBER MATCHING (VECTORIZED)
# ============================================================================

print("\n[STEP 4] TIER 1A - Registration number matching...")

# Filter installations with registration numbers
has_reg = linked[linked['ACC_REG_CLEAN'] != ''].copy()
print(f"  Installations with registration numbers: {len(has_reg):,}")

tier1a_matches = []

if len(has_reg) > 0:
    print("  Matching registration numbers...")
    
    for country in KEEP_COUNTRIES:
        country_data = has_reg[has_reg['ACCOUNT_HOLDER_COUNTRY_CODE'] == country]
        orbis_country = orbis[orbis['Country ISO code'] == country]
        
        if len(country_data) == 0 or len(orbis_country) == 0:
            continue
        
        print(f"    {country}: {len(country_data):,} installations", end='\r')
        
        # Create lookup dict for ORBIS
        orbis_lookup = orbis_country.set_index('BvD ID number')[['NATIONAL_ID_NORMALIZED', 'Company name Latin alphabet']].to_dict('index')
        
        for _, inst in country_data.iterrows():
            reg_num = inst['ACC_REG_CLEAN']
            
            # Check each ORBIS company in this country
            for bvd_id, orbis_data in orbis_lookup.items():
                nat_ids = orbis_data['NATIONAL_ID_NORMALIZED']
                
                # Check if reg_num matches any of the semicolon-separated IDs
                if reg_num in nat_ids.split('; '):
                    tier1a_matches.append({
                        'INSTALLATION_IDENTIFIER': inst['INSTALLATION_IDENTIFIER'],
                        'YEAR': inst['YEAR'],
                        'COUNTRY': inst['COUNTRY'],
                        'BVD_ID': bvd_id,
                        'MATCH_TYPE': 'TIER1A_REG',
                        'MATCH_SCORE': 1.0
                    })
                    break

tier1a_df = pd.DataFrame(tier1a_matches)
print(f"\n  [OK] Tier 1A: {len(tier1a_df):,} matches")

# ============================================================================
# STEP 5: TIER 1B - NAME + POSTAL (VECTORIZED MERGE)
# ============================================================================

print("\n[STEP 5] TIER 1B - Name + Postal code...")

tier1a_inst = set(tier1a_df['INSTALLATION_IDENTIFIER'].unique()) if len(tier1a_df) > 0 else set()
remaining = linked[~linked['INSTALLATION_IDENTIFIER'].isin(tier1a_inst)].copy()

# Create match keys
remaining['MATCH_KEY'] = remaining['ACC_NAME_CLEAN'] + '_' + remaining['ACC_POSTAL_CLEAN'] + '_' + remaining['ACCOUNT_HOLDER_COUNTRY_CODE']
orbis['MATCH_KEY'] = orbis['NAME_CLEAN'] + '_' + orbis['POSTAL_CLEAN'] + '_' + orbis['Country ISO code']

# Merge on match key
tier1b_matches = remaining.merge(
    orbis[['MATCH_KEY', 'BvD ID number']],
    on='MATCH_KEY',
    how='inner'
)[['INSTALLATION_IDENTIFIER', 'YEAR', 'COUNTRY', 'BvD ID number']].copy()

tier1b_matches.columns = ['INSTALLATION_IDENTIFIER', 'YEAR', 'COUNTRY', 'BVD_ID']
tier1b_matches['MATCH_TYPE'] = 'TIER1B_NAME_POSTAL'
tier1b_matches['MATCH_SCORE'] = 1.0

print(f"  [OK] Tier 1B: {len(tier1b_matches):,} matches")

# ============================================================================
# STEP 6: TIER 1C - NAME + CITY (VECTORIZED MERGE)
# ============================================================================

print("\n[STEP 6] TIER 1C - Name + City...")

tier1ab_inst = tier1a_inst | set(tier1b_matches['INSTALLATION_IDENTIFIER'].unique())
remaining = linked[~linked['INSTALLATION_IDENTIFIER'].isin(tier1ab_inst)].copy()

# Create match keys
remaining['MATCH_KEY'] = remaining['ACC_NAME_CLEAN'] + '_' + remaining['ACC_CITY_CLEAN'] + '_' + remaining['ACCOUNT_HOLDER_COUNTRY_CODE']
orbis['MATCH_KEY_CITY'] = orbis['NAME_CLEAN'] + '_' + orbis['CITY_CLEAN'] + '_' + orbis['Country ISO code']

# Merge
tier1c_matches = remaining.merge(
    orbis[['MATCH_KEY_CITY', 'BvD ID number']],
    left_on='MATCH_KEY',
    right_on='MATCH_KEY_CITY',
    how='inner'
)[['INSTALLATION_IDENTIFIER', 'YEAR', 'COUNTRY', 'BvD ID number']].copy()

tier1c_matches.columns = ['INSTALLATION_IDENTIFIER', 'YEAR', 'COUNTRY', 'BVD_ID']
tier1c_matches['MATCH_TYPE'] = 'TIER1C_NAME_CITY'
tier1c_matches['MATCH_SCORE'] = 0.95

print(f"  [OK] Tier 1C: {len(tier1c_matches):,} matches")

# ============================================================================
# STEP 7: TIER 2 - FUZZY MATCHING BY COUNTRY
# ============================================================================

print("\n[STEP 7] TIER 2 - Fuzzy matching by country...")

tier1_all = pd.concat([tier1a_df, tier1b_matches, tier1c_matches], ignore_index=True)
tier1_inst = set(tier1_all['INSTALLATION_IDENTIFIER'].unique())
unmatched = compliance[~compliance['INSTALLATION_IDENTIFIER'].isin(tier1_inst)].copy()

print(f"  Remaining for fuzzy matching: {len(unmatched):,}")

unmatched['INST_NAME_CLEAN'] = unmatched['INSTALLATION_NAME'].apply(clean_name)

def similarity_func(s1, s2):
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(s1, s2) / 100.0
    else:
        return SequenceMatcher(None, s1, s2).ratio()

tier2_matches = []

for country in KEEP_COUNTRIES:
    country_unm = unmatched[unmatched['COUNTRY'] == country]
    orbis_country = orbis[orbis['Country ISO code'] == country]
    
    if len(country_unm) == 0 or len(orbis_country) == 0:
        continue
    
    print(f"  Fuzzy {country}: {len(country_unm):,} vs {len(orbis_country):,}...")
    
    orbis_names = orbis_country['NAME_CLEAN'].tolist()
    orbis_dict = orbis_country.drop_duplicates('NAME_CLEAN').set_index('NAME_CLEAN')['BvD ID number'].to_dict()
    
    for _, inst in country_unm.iterrows():
        best_score = 0
        best_match = None
        
        for orbis_name in orbis_names:
            score = similarity_func(inst['INST_NAME_CLEAN'], orbis_name)
            if score > best_score and score >= SIMILARITY_THRESHOLD:
                best_score = score
                best_match = orbis_name
        
        if best_match:
            tier2_matches.append({
                'INSTALLATION_IDENTIFIER': inst['INSTALLATION_IDENTIFIER'],
                'YEAR': inst['YEAR'],
                'COUNTRY': inst['COUNTRY'],
                'BVD_ID': orbis_dict[best_match],
                'MATCH_TYPE': 'TIER2_FUZZY',
                'MATCH_SCORE': best_score
            })

tier2_df = pd.DataFrame(tier2_matches)
print(f"  [OK] Tier 2: {len(tier2_df):,} matches")

# ============================================================================
# STEP 8: COMBINE AND AGGREGATE
# ============================================================================

print("\n[STEP 8] Combining matches and aggregating...")

all_matches = pd.concat([tier1_all, tier2_df], ignore_index=True)
print(f"  Total matches: {len(all_matches):,}")

# Merge with compliance to get emissions
final = all_matches.merge(
    compliance[['INSTALLATION_IDENTIFIER', 'YEAR', 'TOTAL_VERIFIED_EMISSIONS', 
                'TOTAL_SURRENDERED_ALLOWANCES', 'MAIN_ACTIVITY_TYPE_CODE']],
    on=['INSTALLATION_IDENTIFIER', 'YEAR']
).merge(
    orbis,
    left_on='BVD_ID',
    right_on='BvD ID number'
)

# Aggregate to firm-year
firm_level = final.groupby(['BVD_ID', 'YEAR']).agg({
    'TOTAL_VERIFIED_EMISSIONS': 'sum',
    'TOTAL_SURRENDERED_ALLOWANCES': 'sum',
    'INSTALLATION_IDENTIFIER': 'count',
    'COUNTRY': 'first',
    'Company name Latin alphabet': 'first',
    'NACE Rev. 2, core code (4 digits)': 'first'
}).reset_index()

firm_level.rename(columns={'INSTALLATION_IDENTIFIER': 'NUM_INSTALLATIONS'}, inplace=True)
firm_level['PARTICIPATED'] = firm_level['TOTAL_SURRENDERED_ALLOWANCES'] > 0

print(f"  [OK] Firm-year observations: {len(firm_level):,}")

# NOTE: Steps 9-10 are specific to this study's panel structure.
# Column names reflect Orbis export format used in this analysis (years 2013-2024).
# Users replicating with different Orbis exports may need to adjust column names accordingly.

# ============================================================================
# STEP 9: CREATE PANEL WITH FINANCIAL DATA (INCLUDING AGE)
# ============================================================================

print("\n[STEP 9] Creating panel with year-specific financial data...")

firm_panel = []

# For treated firms, create observations for ALL years 2013-2024
for _, row in firm_level.iterrows():
    bvd_id = row['BVD_ID']
    orbis_firm = orbis[orbis['BvD ID number'] == bvd_id].iloc[0]
    
    # Get participation years for this firm
    firm_years = firm_level[firm_level['BVD_ID'] == bvd_id]['YEAR'].unique()
    
    # Create observations for all years 2013-2024
    for year in range(2013, 2025):
        # Firm participated in this year if:
        # 1. Year is 2021-2024 AND
        # 2. Firm appears in firm_level for that year
        participated = (year >= 2021) and (year in firm_years)
        
        # Get emissions/allowances if participated, otherwise 0
        if participated:
            year_data = firm_level[(firm_level['BVD_ID'] == bvd_id) & (firm_level['YEAR'] == year)].iloc[0]
            emissions = year_data['TOTAL_VERIFIED_EMISSIONS']
            allowances = year_data['TOTAL_SURRENDERED_ALLOWANCES']
            n_inst = year_data['NUM_INSTALLATIONS']
        else:
            emissions = 0
            allowances = 0
            n_inst = 0
        
        firm_row = {
            'BVD_ID': bvd_id,
            'YEAR': year,
            'Company_Name': orbis_firm['Company name Latin alphabet'],
            'Country': orbis_firm['Country ISO code'],
            'NACE_Code': orbis_firm['NACE Rev. 2, core code (4 digits)'],
            'Date_of_Incorporation': orbis_firm['Date of incorporation'],
            'TREATED_FIRM': 1,
            'PARTICIPATED': participated,
            'Total_Emissions': emissions,
            'Total_Allowances': allowances,
            'Num_Installations': n_inst,
            'Total_Assets': orbis_firm[f'Total assets\nth EUR {year}'],
            'Non_Current_Liabilities': orbis_firm[f'Non-current liabilities\nth EUR {year}'],
            'ROA': orbis_firm[f'ROA using Net income\n{year}'],
            'Operating_Revenue': orbis_firm[f'Operating revenue (Turnover)\nth EUR {year}'],
            'Investments': orbis_firm[f'Investments\nth EUR {year}'],
            'Tangible_Fixed_Assets': orbis_firm[f'Tangible fixed assets\nth EUR {year}'],
            'Current_Liabilities': orbis_firm[f'Current liabilities\nth EUR {year}'],
            'Liquidity_Ratio': orbis_firm[f'Liquidity ratio\n{year}'],
            'Tobins_Q': orbis_firm.get(f"Market capitalisation / Total assets (Tobin's Q)\n{year}", np.nan)
        }
        firm_panel.append(firm_row)

firm_panel_df = pd.DataFrame(firm_panel)

# Remove duplicate firm-years
firm_panel_df = firm_panel_df.drop_duplicates(subset=['BVD_ID', 'YEAR'], keep='first')

print(f"  [OK] Treated firm panel: {len(firm_panel_df):,} observations")

# ============================================================================
# STEP 10: CREATE CONTROL GROUP (INCLUDING AGE)
# ============================================================================

print("\n[STEP 10] Creating control group...")

matched_bvd = set(firm_panel_df['BVD_ID'].unique())
control = orbis[~orbis['BvD ID number'].isin(matched_bvd) & 
                orbis['Country ISO code'].isin(KEEP_COUNTRIES)]

print(f"  Control firms: {len(control):,}")

control_panel = []
# Include all years 2013-2024 for control group
for year in range(2013, 2025):
    for _, firm in control.iterrows():
        control_panel.append({
            'BVD_ID': firm['BvD ID number'],
            'YEAR': year,
            'Company_Name': firm['Company name Latin alphabet'],
            'Country': firm['Country ISO code'],
            'NACE_Code': firm['NACE Rev. 2, core code (4 digits)'],
            'Date_of_Incorporation': firm['Date of incorporation'],
            'TREATED_FIRM': 0,
            'PARTICIPATED': False,
            'Total_Emissions': 0,
            'Total_Allowances': 0,
            'Num_Installations': 0,
            'Total_Assets': firm[f'Total assets\nth EUR {year}'],
            'Non_Current_Liabilities': firm[f'Non-current liabilities\nth EUR {year}'],
            'ROA': firm[f'ROA using Net income\n{year}'],
            'Operating_Revenue': firm[f'Operating revenue (Turnover)\nth EUR {year}'],
            'Investments': firm[f'Investments\nth EUR {year}'],
            'Tangible_Fixed_Assets': firm[f'Tangible fixed assets\nth EUR {year}'],
            'Current_Liabilities': firm[f'Current liabilities\nth EUR {year}'],
            'Liquidity_Ratio': firm[f'Liquidity ratio\n{year}'],
            'Tobins_Q': firm.get(f"Market capitalisation / Total assets (Tobin's Q)\n{year}", np.nan)
        })

control_panel_df = pd.DataFrame(control_panel)

print(f"  [OK] Control panel: {len(control_panel_df):,} observations")

# ============================================================================
# STEP 11: SAVE
# ============================================================================

print("\n[STEP 11] Saving...")

final_dataset = pd.concat([firm_panel_df, control_panel_df], ignore_index=True)
final_dataset.to_csv('matched_euets_orbis_final.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] matched_euets_orbis_final.csv ({len(final_dataset):,} rows)")

all_matches.to_csv('matching_details_tiered.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] matching_details_tiered.csv")

print("\nMATCH RATES BY TIER")
print(f"Tier 1A attempted: {len(has_reg):,}")
print(f"Tier 1A matched: {len(tier1a_df):,}")
print(f"Tier 1A match rate: {len(tier1a_df)/len(has_reg)*100:.1f}%")

print(f"\nTier 1B attempted: {len(remaining[~remaining['INSTALLATION_IDENTIFIER'].isin(tier1a_inst)]):,}")
print(f"Tier 1B matched: {len(tier1b_matches):,}")

print(f"\nTier 1C attempted: {len(linked[~linked['INSTALLATION_IDENTIFIER'].isin(tier1ab_inst)]):,}")
print(f"Tier 1C matched: {len(tier1c_matches):,}")

print(f"\nTier 2 attempted: {len(unmatched):,}")
print(f"Tier 2 matched: {len(tier2_df):,}")
print(f"Tier 2 match rate: {len(tier2_df)/len(unmatched)*100:.1f}%")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTotal installations: {compliance['INSTALLATION_IDENTIFIER'].nunique():,}")
print(f"Matched: {all_matches['INSTALLATION_IDENTIFIER'].nunique():,}")
print(f"Match rate: {all_matches['INSTALLATION_IDENTIFIER'].nunique() / compliance['INSTALLATION_IDENTIFIER'].nunique() * 100:.1f}%")
print(f"\nTier 1A (Reg #): {len(tier1a_df):,}")
print(f"Tier 1B (Name+Postal): {len(tier1b_matches):,}")
print(f"Tier 1C (Name+City): {len(tier1c_matches):,}")
print(f"Tier 2 (Fuzzy): {len(tier2_df):,}")
print(f"\nTreated firms: {final_dataset[final_dataset['TREATED_FIRM']==1]['BVD_ID'].nunique():,}")
print(f"Control firms: {final_dataset[final_dataset['TREATED_FIRM']==0]['BVD_ID'].nunique():,}")
print(f"\nYears: 2013-2024")
print("=" * 80)
print("[OK] COMPLETE!")
print("=" * 80)
