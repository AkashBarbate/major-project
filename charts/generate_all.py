"""
generate_all.py
---------------
Master script that runs ALL chart generation scripts.
Generates every chart, table, and figure for the project report.

Run: python charts/generate_all.py

All outputs are saved to: charts/output/
"""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUT_DIR = 'charts/output'
os.makedirs(OUT_DIR, exist_ok=True)

def run_module(name, module_func):
    """Run a chart module with timing."""
    start = time.time()
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    try:
        module_func()
        elapsed = time.time() - start
        print(f"  [OK] {name} completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"  [FAIL] {name} FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    total_start = time.time()

    print("""
============================================================
  CHART & TABLE GENERATOR - Minor Project Report
  Generating all visualizations for:
  Multi-Label News Classification with Credibility & XAI
============================================================
    """)

    # 1. Dataset Analysis (Chapter 2)
    from dataset_analysis import (
        chart_2_1_raw_distribution, chart_2_2_preprocessed_distribution,
        chart_2_3_domain_capping, chart_2_4_source_distribution,
        chart_2_5_word_count_distribution, chart_2_6_imbalance_progression,
        chart_2_7_data_funnel, table_2_4_preprocessing_steps
    )
    run_module("Dataset Analysis", lambda: [
        chart_2_1_raw_distribution(), chart_2_2_preprocessed_distribution(),
        chart_2_3_domain_capping(), chart_2_4_source_distribution(),
        chart_2_5_word_count_distribution(), chart_2_6_imbalance_progression(),
        chart_2_7_data_funnel(), table_2_4_preprocessing_steps()
    ])

    # 2. Training Metrics (Chapter 4)
    from training_metrics import (
        chart_4_4_all_baseline_metrics, chart_4_5_dual_axis,
        chart_4_6_model_comparison, table_4_3_baseline_metrics,
        table_4_4_architecture_comparison
    )
    run_module("Training Metrics", lambda: [
        chart_4_4_all_baseline_metrics(), chart_4_5_dual_axis(),
        chart_4_6_model_comparison(), table_4_3_baseline_metrics(),
        table_4_4_architecture_comparison()
    ])

    # 3. Threshold & Calibration (Chapters 5-6)
    from threshold_calibration import (
        chart_5_1_thresholds, chart_5_2_sensitivity,
        table_5_1_thresholds, chart_6_1_reliability_diagram,
        chart_6_2_ece_comparison
    )
    run_module("Threshold & Calibration", lambda: [
        chart_5_1_thresholds(), chart_5_2_sensitivity(),
        table_5_1_thresholds(), chart_6_1_reliability_diagram(),
        chart_6_2_ece_comparison()
    ])

    # 4. Credibility Analysis (Chapter 7)
    from credibility_analysis import (
        chart_7_1_credibility_distribution, chart_7_2_feature_importance,
        chart_7_4_radar_chart, chart_7_6_correlation_heatmap,
        table_7_1_features, table_7_3_domains
    )
    run_module("Credibility Analysis", lambda: [
        chart_7_1_credibility_distribution(), chart_7_2_feature_importance(),
        chart_7_4_radar_chart(), chart_7_6_correlation_heatmap(),
        table_7_1_features(), table_7_3_domains()
    ])

    # 5. XAI & Results (Chapters 8, 10)
    from xai_results import (
        fig_8_1_xai_keyword_impact, fig_8_2_xai_multi_label,
        table_8_1_xai_methods, chart_10_1_roc_curves,
        chart_10_2_pr_curves, chart_10_5_error_analysis,
        chart_10_6_label_count, table_10_2_per_category
    )
    run_module("XAI & Results", lambda: [
        fig_8_1_xai_keyword_impact(), fig_8_2_xai_multi_label(),
        table_8_1_xai_methods(), chart_10_1_roc_curves(),
        chart_10_2_pr_curves(), chart_10_5_error_analysis(),
        chart_10_6_label_count(), table_10_2_per_category()
    ])

    # 6. Architecture & System (Chapters 1, 3, 9)
    from system_architecture import (
        fig_1_1_architecture, fig_1_3_category_consolidation,
        chart_3_1_label_distribution, chart_3_3_cooccurrence,
        table_3_1_labeling_models, table_9_1_api_endpoints
    )
    run_module("Architecture & System", lambda: [
        fig_1_1_architecture(), fig_1_3_category_consolidation(),
        chart_3_1_label_distribution(), chart_3_3_cooccurrence(),
        table_3_1_labeling_models(), table_9_1_api_endpoints()
    ])

    # ── Summary ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start

    output_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.png')]
    charts = [f for f in output_files if f.startswith('chart_')]
    tables = [f for f in output_files if f.startswith('table_')]
    figures = [f for f in output_files if f.startswith('fig_')]

    print(f"""
============================================================
  GENERATION COMPLETE
============================================================
  Charts generated:  {len(charts)}
  Tables generated:  {len(tables)}
  Figures generated: {len(figures)}
  Total files:       {len(output_files)}
  Time elapsed:      {total_elapsed:.1f}s
  Output directory:  {OUT_DIR}
============================================================
    """)

    print("Generated files:")
    for f in sorted(output_files):
        size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f:<50} {size_kb:.0f} KB")
