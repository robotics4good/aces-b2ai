"""
Parse pediatric multi-select condition fields into binary columns.

Pediatric medical conditions are stored in 9 multi-select fields as comma-separated strings:
    peds_mc_breathing_conditions: "asthma, chronic_cough"
    peds_mc_hearing_loss: "hearing_loss, ear tubes"
    peds_mc_psych_disorders: "anxiety disorder, adhd"
    ...

This script expands these into binary columns (one per unique condition).

Output: pediatric_conditions_expanded.tsv with 30-50 binary condition columns

Usage:
    python scripts/parse_pediatric_conditions.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def parse_multi_select_field(value_str):
    """Parse comma-separated condition string"""
    if pd.isna(value_str):
        return []
    return [c.strip() for c in str(value_str).split(',')]


def main():
    # Load pediatric medical conditions
    input_path = 'phenotype/pediatric/pediatric_medical_conditions.tsv'
    output_path = 'phenotype/pediatric/pediatric_conditions_expanded.tsv'
    report_path = 'pediatric_condition_prevalence_report.csv'

    print(f"Loading pediatric medical conditions from {input_path}...")
    df = pd.read_csv(input_path, sep='\t')

    print(f"Loaded {len(df)} participants")

    # Multi-select fields to parse
    multi_select_fields = [
        'peds_mc_breathing_conditions',
        'peds_mc_hearing_loss',
        'peds_mc_voice_disorders',
        'peds_mc_psych_disorders',
        'peds_mc_conditions',
        'peds_mc_chronic_conditions',
        'peds_mc_neurological_disorders',
        'peds_mc_surgical_history',
        'peds_mc_throat_surgical_history',
    ]

    # Collect all unique conditions across multi-select fields
    all_conditions = set()
    for field in multi_select_fields:
        if field not in df.columns:
            print(f"⚠️  Warning: Field '{field}' not found in TSV, skipping")
            continue

        for value in df[field].dropna():
            conditions = parse_multi_select_field(value)
            all_conditions.update(conditions)

    print(f"\nFound {len(all_conditions)} unique conditions across multi-select fields:")
    for cond in sorted(all_conditions)[:10]:
        print(f"  - {cond}")
    if len(all_conditions) > 10:
        print(f"  ... and {len(all_conditions) - 10} more")

    # Create binary columns for each condition
    expanded_df = pd.DataFrame({'participant_id': df['participant_id']})

    for condition in sorted(all_conditions):
        col_name = f'has_{condition.replace(" ", "_").replace("-", "_")}'
        expanded_df[col_name] = 0

        # Check each multi-select field
        for field in multi_select_fields:
            if field not in df.columns:
                continue

            for idx, value in enumerate(df[field]):
                if pd.notna(value) and condition in parse_multi_select_field(value):
                    expanded_df.loc[idx, col_name] = 1

    # Add binary yes/no fields (not multi-select)
    binary_fields = {
        'had_allergies': 'peds_mc_allergies',
        'had_speech_therapy': 'peds_mc_a_therapy',
        'had_tonsillectomy': 'peds_mc_tonsillectomy',
        'had_previous_hospitalization': 'peds_mc_previous_hospitalization',
    }

    for new_col, orig_col in binary_fields.items():
        if orig_col in df.columns:
            expanded_df[new_col] = (df[orig_col] == 'yes').astype(int)
        else:
            print(f"⚠️  Warning: Binary field '{orig_col}' not found, skipping")

    # Save expanded TSV
    expanded_df.to_csv(output_path, sep='\t', index=False)
    print(f"\n✅ Saved expanded conditions to {output_path}")
    print(f"   Shape: {expanded_df.shape} ({len(expanded_df)} participants × {len(expanded_df.columns)-1} conditions)")

    # Generate prevalence report
    condition_cols = [c for c in expanded_df.columns if c != 'participant_id']
    prevalence = expanded_df[condition_cols].mean().sort_values(ascending=False)
    prevalence_df = pd.DataFrame({
        'condition': prevalence.index,
        'prevalence': prevalence.values,
        'count': (prevalence.values * len(expanded_df)).astype(int)
    })

    prevalence_df.to_csv(report_path, index=False)
    print(f"\n✅ Saved prevalence report to {report_path}")

    print("\nTop 15 most prevalent conditions:")
    print(prevalence_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
