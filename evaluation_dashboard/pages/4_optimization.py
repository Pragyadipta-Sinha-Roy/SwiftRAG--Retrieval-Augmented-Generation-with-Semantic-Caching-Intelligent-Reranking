"""
Threshold Optimization & Model Comparison
Find the optimal configuration for production deployment
"""
from sentence_transformers import SentenceTransformer
from cachelab.evaluate.evaluation_result import EvaluationResult
from cachelab.evaluate.evaluatable_cache import EvaluatableCache
from cachelab.evaluate.cache_evaluator import CacheEvaluator
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time


st.set_page_config(page_title="Optimization", page_icon="⚙️", layout="wide")

st.title("⚙️ Threshold Optimization & Model Comparison")
st.markdown("### Find the optimal configuration for production deployment")

st.markdown("---")

# Check if data is loaded
if st.session_state.get('ground_truth_df') is None or st.session_state.get('test_df') is None:
    st.warning("⚠️ Please load data first!")
    if st.button("Go to Data Page"):
        st.switch_page("pages/2_data.py")
    st.stop()

ground_truth_df = st.session_state.ground_truth_df
test_df = st.session_state.test_df

# Introduction
st.info("""
**Goal:** Find the optimal threshold and embedding model for your production semantic cache.

This page helps you:
- 🎯 Systematically test multiple thresholds
- 📊 Compare embedding models (sentence-transformers)
- 📈 Visualize precision-recall trade-offs
- 💡 Get data-driven recommendations for deployment
""")

st.markdown("---")

# Configuration Section
st.markdown("## 🔧 Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📐 Threshold Range")

    threshold_min = st.number_input(
        "Minimum Threshold",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05,
        help="Start of threshold range to test"
    )

    threshold_max = st.number_input(
        "Maximum Threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="End of threshold range to test"
    )

    threshold_step = st.number_input(
        "Step Size",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Increment between threshold values"
    )

    # Generate threshold list
    thresholds = list(
        np.arange(threshold_min, threshold_max + threshold_step, threshold_step))
    st.info(
        f"Will test **{len(thresholds)} thresholds**: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")

with col2:
    st.markdown("### 🤖 Models to Compare")

    st.markdown("**Sentence Transformers (Open Source):**")
    test_minilm = st.checkbox("all-MiniLM-L6-v2 (Fast, 384d)", value=True)
    test_mpnet = st.checkbox("all-mpnet-base-v2 (Balanced, 768d)", value=True)
    test_paraphrase = st.checkbox(
        "paraphrase-MiniLM-L6-v2 (Paraphrase-optimized, 384d)", value=False)
    test_multilingual = st.checkbox(
        "paraphrase-multilingual-MiniLM-L12-v2 (Multilingual, 384d)", value=False)

# Build model list
selected_models = []
if test_minilm:
    selected_models.append(
        ("all-MiniLM-L6-v2", "MiniLM-L6 (Fast)", "sentence-transformers"))
if test_mpnet:
    selected_models.append(
        ("all-mpnet-base-v2", "MPNet (Balanced)", "sentence-transformers"))
if test_paraphrase:
    selected_models.append(
        ("paraphrase-MiniLM-L6-v2", "Paraphrase-MiniLM", "sentence-transformers"))
if test_multilingual:
    selected_models.append(("paraphrase-multilingual-MiniLM-L12-v2",
                           "Multilingual-MiniLM", "sentence-transformers"))
st.info(f"**{len(selected_models)} models** selected for comparison")

st.markdown("---")

# Run optimization
run_optimization = st.button(
    "🚀 Run Optimization", type="primary", use_container_width=True)

# All modules are implemented
if run_optimization:
    # Initialize results storage in session state
    st.session_state['all_threshold_results'] = {}
    st.session_state['model_comparison_results'] = None
    st.session_state['optimization_config'] = {
        'thresholds': thresholds,
        'selected_models': selected_models
    }

    st.markdown("---")
    st.markdown("## 📊 Running Optimization...")

    # Create evaluator once (will be used for all models)
    evaluator = CacheEvaluator(ground_truth_df, test_df)
    qa_pairs = list(
        zip(ground_truth_df["question"], ground_truth_df["answer"]))

    # Run threshold sweep for ALL selected models
    progress_container = st.empty()

    for model_idx, (model_name, model_display, model_type) in enumerate(selected_models):
        progress_container.info(
            f"Testing {model_display} ({model_idx + 1}/{len(selected_models)})...")

        with st.spinner(f"Initializing {model_display}..."):
            # Load embedding model
            encoder = SentenceTransformer(model_name)

            # Create evaluatable cache
            cache = EvaluatableCache(encoder, distance_threshold=0.3)
            cache.add_many(qa_pairs)

        # Real threshold evaluation
        threshold_results = []

        for i, threshold in enumerate(thresholds):
            # Real evaluation with threshold override
            def check_fn(q, t=threshold):
                return cache.check(q, threshold_override=t)
            result = evaluator.evaluate(check_fn)

            threshold_results.append({
                'threshold': threshold,
                'precision': result.precision,
                'recall': result.recall,
                'f1': result.f1_score,
                'accuracy': result.accuracy,
                'hit_rate': result.hit_rate
            })

        # Store results for this model
        st.session_state['all_threshold_results'][model_display] = threshold_results

    progress_container.success("✅ All threshold sweeps complete!")

# Display results if they exist
if st.session_state.get('all_threshold_results'):

    st.markdown("---")
    st.markdown("## 📊 Optimization Results")

    # Retrieve stored data
    all_threshold_results = st.session_state['all_threshold_results']
    config = st.session_state.get('optimization_config', {})
    thresholds = config.get('thresholds', thresholds)
    selected_models = config.get('selected_models', selected_models)

    # Tab structure for results
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Threshold Analysis",
        "🤖 Model Comparison",
        "🎯 Optimal Configuration",
        "📋 Detailed Metrics"
    ])

    with tab1:
        st.markdown("### Threshold Sweep Results")

        # Select model for threshold analysis
        if len(all_threshold_results) > 0:
            model_names = list(all_threshold_results.keys())

            selected_model_display = st.selectbox(
                "Select model for threshold analysis",
                model_names,
                key="threshold_model_selector"
            )

            # Get results for selected model
            threshold_results = all_threshold_results[selected_model_display]

            # Find optimal thresholds
            best_by_f1 = max(threshold_results, key=lambda x: x['f1'])
            best_by_precision = max(
                threshold_results, key=lambda x: x['precision'])
            best_by_recall = max(threshold_results, key=lambda x: x['recall'])

            # Display optimal thresholds
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Best F1 Score",
                    f"{best_by_f1['f1']:.2%}",
                    f"Threshold: {best_by_f1['threshold']:.2f}"
                )

            with col2:
                st.metric(
                    "Best Precision",
                    f"{best_by_precision['precision']:.2%}",
                    f"Threshold: {best_by_precision['threshold']:.2f}"
                )

            with col3:
                st.metric(
                    "Best Recall",
                    f"{best_by_recall['recall']:.2%}",
                    f"Threshold: {best_by_recall['threshold']:.2f}"
                )

            st.markdown("---")

            # Visualization 1: Metrics vs Threshold
            st.markdown("#### Metrics vs Threshold")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=[r['threshold'] for r in threshold_results],
                y=[r['precision'] for r in threshold_results],
                mode='lines+markers',
                name='Precision',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))

            fig.add_trace(go.Scatter(
                x=[r['threshold'] for r in threshold_results],
                y=[r['recall'] for r in threshold_results],
                mode='lines+markers',
                name='Recall',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))

            fig.add_trace(go.Scatter(
                x=[r['threshold'] for r in threshold_results],
                y=[r['f1'] for r in threshold_results],
                mode='lines+markers',
                name='F1 Score',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))

            # Add vertical line for best F1
            fig.add_vline(
                x=best_by_f1['threshold'],
                line_dash="dash",
                line_color="green",
                annotation_text=f"Best F1: {best_by_f1['threshold']:.2f}",
                annotation_position="top"
            )

            fig.update_layout(
                title=f"Threshold Analysis - {selected_model_display}",
                xaxis_title="Distance Threshold",
                yaxis_title="Score",
                hovermode='x unified',
                height=500,
                yaxis=dict(range=[0, 1.05])
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Visualization 2: Precision-Recall Curve
            st.markdown("#### Precision-Recall Curve")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=[r['recall'] for r in threshold_results],
                y=[r['precision'] for r in threshold_results],
                mode='lines+markers+text',
                text=[f"{r['threshold']:.2f}" for r in threshold_results],
                textposition="top center",
                marker=dict(
                    size=10,
                    color=[r['f1'] for r in threshold_results],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="F1 Score")
                ),
                line=dict(color='blue', width=2),
                hovertemplate='<b>Threshold: %{text}</b><br>Recall: %{x:.2%}<br>Precision: %{y:.2%}<extra></extra>'
            ))

            fig.update_layout(
                title="Precision-Recall Trade-off",
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=500,
                xaxis=dict(range=[0, 1.05]),
                yaxis=dict(range=[0, 1.05])
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Visualization 3: Hit Rate & Accuracy
            st.markdown("#### Hit Rate & Accuracy vs Threshold")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=[r['threshold'] for r in threshold_results],
                y=[r['hit_rate'] for r in threshold_results],
                mode='lines+markers',
                name='Hit Rate',
                line=dict(color='magenta', width=3),
                marker=dict(size=8)
            ))

            fig.add_trace(go.Scatter(
                x=[r['threshold'] for r in threshold_results],
                y=[r['accuracy'] for r in threshold_results],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='cyan', width=3),
                marker=dict(size=8)
            ))

            fig.update_layout(
                title="Hit Rate & Accuracy Analysis",
                xaxis_title="Distance Threshold",
                yaxis_title="Rate",
                hovermode='x unified',
                height=400,
                yaxis=dict(range=[0, 1.05])
            )

            st.plotly_chart(fig, use_container_width=True)

            # Threshold recommendation
            st.markdown("---")
            st.markdown("#### 💡 Threshold Recommendations")

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"""
                **For Maximum F1 Score (Balanced):**
                - Threshold: `{best_by_f1['threshold']:.2f}`
                - Precision: {best_by_f1['precision']:.2%}
                - Recall: {best_by_f1['recall']:.2%}
                - F1: {best_by_f1['f1']:.2%}
                
                **Use when:** You want balanced performance
                """)

            with col2:
                st.warning(f"""
                **For Maximum Precision (Minimize False Positives):**
                - Threshold: `{best_by_precision['threshold']:.2f}`
                - Precision: {best_by_precision['precision']:.2%}
                - Recall: {best_by_precision['recall']:.2%}
                - F1: {best_by_precision['f1']:.2%}
                
                **Use when:** Wrong answers are very costly
                """)

            # Results table
            st.markdown("---")
            st.markdown("#### 📋 Complete Results Table")

            results_df = pd.DataFrame(threshold_results)
            st.dataframe(
                results_df.style.format({
                    'threshold': '{:.2f}',
                    'precision': '{:.2%}',
                    'recall': '{:.2%}',
                    'f1': '{:.2%}',
                    'accuracy': '{:.2%}',
                    'hit_rate': '{:.2%}'
                }).background_gradient(subset=['f1'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
        else:
            st.warning(
                "No threshold results available. Please run optimization first.")

    with tab2:
        st.markdown("### Model Performance Comparison")

        if len(selected_models) > 1:

            # **FIX #2: Add threshold selector for model comparison**
            comparison_threshold = st.slider(
                "Select threshold for model comparison",
                min_value=float(thresholds[0]),
                max_value=float(thresholds[-1]),
                value=0.30,
                step=float(threshold_step),
                key="comparison_threshold_selector"
            )

            st.info(
                f"Comparing {len(selected_models)} models at threshold = {comparison_threshold:.2f}")

            # Extract comparison results from threshold sweep data
            model_comparison_results = []

            for model_display in all_threshold_results.keys():
                threshold_results = all_threshold_results[model_display]

                # Find result closest to selected threshold
                closest_result = min(threshold_results,
                                     key=lambda x: abs(x['threshold'] - comparison_threshold))

                # Find model info
                model_info = next(
                    (m for m in selected_models if m[1] == model_display), None)
                if model_info:
                    model_name, _, model_type = model_info

                    model_comparison_results.append({
                        'model': model_display,
                        'model_name': model_name,
                        'model_type': model_type,
                        'f1': closest_result['f1'],
                        'precision': closest_result['precision'],
                        'recall': closest_result['recall'],
                        'threshold': closest_result['threshold']
                    })

            # Best model
            best_model = max(model_comparison_results, key=lambda x: x['f1'])

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "🏆 Best F1 Score",
                    best_model['model'],
                    f"{best_model['f1']:.2%}"
                )

            with col2:
                st.metric(
                    "📊 At Threshold",
                    f"{comparison_threshold:.2f}",
                    ""
                )

            st.markdown("---")

            # Model comparison table
            st.markdown("#### 📊 Model Performance Table")

            comparison_df = pd.DataFrame(model_comparison_results)
            display_df = comparison_df[[
                'model', 'f1', 'precision', 'recall', 'threshold']].copy()
            display_df.columns = ['Model', 'F1 Score',
                                  'Precision', 'Recall', 'Threshold']

            st.dataframe(
                display_df.style.format({
                    'F1 Score': '{:.2%}',
                    'Precision': '{:.2%}',
                    'Recall': '{:.2%}',
                    'Threshold': '{:.2f}'
                }).background_gradient(subset=['F1 Score'], cmap='RdYlGn'),
                use_container_width=True
            )

            st.markdown("---")

            # Model comparison visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### F1 Score Comparison")

                fig = go.Figure(data=[
                    go.Bar(
                        x=[r['model'] for r in model_comparison_results],
                        y=[r['f1'] for r in model_comparison_results],
                        marker_color=[r['f1']
                                      for r in model_comparison_results],
                        marker=dict(colorscale='RdYlGn', showscale=False),
                        text=[f"{r['f1']:.2%}" for r in model_comparison_results],
                        textposition='auto'
                    )
                ])

                fig.update_layout(
                    xaxis_title="Model",
                    yaxis_title="F1 Score",
                    height=400,
                    yaxis=dict(range=[0, 1.05])
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Precision vs Recall")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=[r['recall'] for r in model_comparison_results],
                    y=[r['precision'] for r in model_comparison_results],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color=[r['f1'] for r in model_comparison_results],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="F1 Score")
                    ),
                    text=[r['model'] for r in model_comparison_results],
                    textposition="top center",
                    hovertemplate='<b>%{text}</b><br>Recall: %{x:.2%}<br>Precision: %{y:.2%}<extra></extra>'
                ))

                fig.update_layout(
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=400,
                    xaxis=dict(range=[0, 1.05]),
                    yaxis=dict(range=[0, 1.05])
                )

                st.plotly_chart(fig, use_container_width=True)

            # Model recommendations
            st.markdown("---")
            st.markdown("#### 💡 Model Recommendations")

            st.success(f"""
            **🏆 Recommended for Production: {best_model['model']}**
            
            - **F1 Score:** {best_model['f1']:.2%}
            - **Precision:** {best_model['precision']:.2%}
            - **Recall:** {best_model['recall']:.2%}
            - **Threshold:** {best_model['threshold']:.2f}
            
            This model provides the best balance of accuracy and performance for your dataset at threshold {comparison_threshold:.2f}.
            """)

        elif len(selected_models) == 1:
            st.warning("Please select at least 2 models to compare")
        else:
            st.warning("Please select models to compare")

    with tab3:
        st.markdown("### 🎯 Optimal Configuration")

        # Get best configuration from all results
        best_overall_f1 = 0
        best_config = None

        for model_display, threshold_results in all_threshold_results.items():
            best_for_model = max(threshold_results, key=lambda x: x['f1'])
            if best_for_model['f1'] > best_overall_f1:
                best_overall_f1 = best_for_model['f1']
                best_config = {
                    'model_display': model_display,
                    'threshold': best_for_model['threshold'],
                    'f1': best_for_model['f1'],
                    'precision': best_for_model['precision'],
                    'recall': best_for_model['recall']
                }

        if best_config:
            # Find model name
            model_info = next(
                (m for m in selected_models if m[1] == best_config['model_display']), None)
            model_name = model_info[0] if model_info else 'all-MiniLM-L6-v2'
            model_type = model_info[2] if model_info else 'sentence-transformers'

            st.success(
                "Based on your optimization results, here's the recommended production configuration:")

            # Display configuration card
            st.markdown("### 📋 Production Configuration")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.code(f"""
# Optimal Semantic Cache Configuration

EMBEDDING_MODEL = "{model_name}"
DISTANCE_THRESHOLD = {best_config['threshold']:.2f}

# Expected Performance
# - Precision: {best_config['precision']:.2%}
# - Recall: {best_config['recall']:.2%}
# - F1 Score: {best_config['f1']:.2%}
                """, language="python")

            with col2:
                st.metric("Expected F1", f"{best_config['f1']:.2%}")
                st.metric("Precision", f"{best_config['precision']:.2%}")
                st.metric("Recall", f"{best_config['recall']:.2%}")

            st.markdown("---")

            # Implementation example
            st.markdown("### 💻 Implementation Example")

            st.code(f"""
from sentence_transformers import SentenceTransformer
from your_cache_module import SemanticCache

# Initialize with optimal configuration
encoder = SentenceTransformer("{model_name}")
cache = SemanticCache(
    encoder=encoder,
    distance_threshold={best_config['threshold']:.2f}
)

# Hydrate cache with your FAQ data
qa_pairs = [
    ("How do I get a refund?", "Visit orders page..."),
    ("Can I reset my password?", "Click Forgot Password..."),
    # ... more pairs
]
cache.add_many(qa_pairs)

# Use in production
def get_answer(user_query: str) -> str:
    result = cache.check(user_query)
    
    if result:
        # Cache hit - return cached answer
        return result['answer']
    else:
        # Cache miss - call LLM
        return call_llm_api(user_query)
            """, language="python")

            st.markdown("---")

            # Deployment checklist
            st.markdown("### ✅ Deployment Checklist")

            checklist = [
                ("Download and save your embedding model", False),
                ("Set up cache hydration pipeline", False),
                ("Implement cache-first query logic", False),
                ("Add monitoring for hit rate", False),
                ("Set up A/B testing (optional)", False),
                ("Configure cache refresh strategy", False)
            ]

            for item, _ in checklist:
                st.checkbox(item, key=f"checklist_{item}")

            st.markdown("---")

            # Export configuration
            st.markdown("### 💾 Export Configuration")

            col1, col2 = st.columns(2)

            with col1:
                config_text = f"""# Semantic Cache Configuration
# Generated by Cache Optimizer Dashboard
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

[model]
name = "{model_name}"
type = "{model_type}"

[cache]
distance_threshold = {best_config['threshold']:.2f}

[performance]
expected_precision = {best_config['precision']:.2%}
expected_recall = {best_config['recall']:.2%}
expected_f1 = {best_config['f1']:.2%}

[dataset]
faq_entries = {len(ground_truth_df)}
test_queries = {len(test_df)}
"""
                st.download_button(
                    "📥 Download Config (TOML)",
                    config_text,
                    "semantic_cache_config.toml",
                    "text/plain",
                    use_container_width=True
                )

            with col2:
                json_config = f"""{{
  "model": {{
    "name": "{model_name}",
    "type": "{model_type}"
  }},
  "cache": {{
    "distance_threshold": {best_config['threshold']:.2f}
  }},
  "performance": {{
    "expected_precision": {best_config['precision']:.4f},
    "expected_recall": {best_config['recall']:.4f},
    "expected_f1": {best_config['f1']:.4f}
  }}
}}"""
                st.download_button(
                    "📥 Download Config (JSON)",
                    json_config,
                    "semantic_cache_config.json",
                    "application/json",
                    use_container_width=True
                )
        else:
            st.info("Run optimization to see recommended configuration")

    with tab4:
        st.markdown("### 📋 Detailed Metrics & Analysis")

        # **FIX #3: Show detailed metrics for selected model from Tab 1**
        if all_threshold_results:
            # Use the same model selector as Tab 1
            model_names = list(all_threshold_results.keys())

            detailed_model_display = st.selectbox(
                "Select model for detailed metrics",
                model_names,
                key="detailed_metrics_model_selector"
            )

            threshold_results = all_threshold_results[detailed_model_display]

            st.markdown(
                f"#### Complete Threshold Analysis for {detailed_model_display}")

            # Full results table with all metrics
            detailed_df = pd.DataFrame(threshold_results)

            st.dataframe(
                detailed_df.style.format({
                    'threshold': '{:.3f}',
                    'precision': '{:.3f}',
                    'recall': '{:.3f}',
                    'f1': '{:.3f}',
                    'accuracy': '{:.3f}',
                    'hit_rate': '{:.3f}'
                }),
                use_container_width=True,
                height=500
            )

            # Download detailed results
            csv = detailed_df.to_csv(index=False)
            st.download_button(
                f"📥 Download Detailed Results for {detailed_model_display} (CSV)",
                csv,
                f"threshold_analysis_{detailed_model_display.replace(' ', '_')}.csv",
                "text/csv",
                key=f"download_detailed_{detailed_model_display}"
            )

            st.markdown("---")

            # Statistical summary
            st.markdown("#### Statistical Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Precision**")
                st.write(f"Mean: {detailed_df['precision'].mean():.2%}")
                st.write(f"Std: {detailed_df['precision'].std():.2%}")
                st.write(f"Min: {detailed_df['precision'].min():.2%}")
                st.write(f"Max: {detailed_df['precision'].max():.2%}")

            with col2:
                st.markdown("**Recall**")
                st.write(f"Mean: {detailed_df['recall'].mean():.2%}")
                st.write(f"Std: {detailed_df['recall'].std():.2%}")
                st.write(f"Min: {detailed_df['recall'].min():.2%}")
                st.write(f"Max: {detailed_df['recall'].max():.2%}")

            with col3:
                st.markdown("**F1 Score**")
                st.write(f"Mean: {detailed_df['f1'].mean():.2%}")
                st.write(f"Std: {detailed_df['f1'].std():.2%}")
                st.write(f"Min: {detailed_df['f1'].min():.2%}")
                st.write(f"Max: {detailed_df['f1'].max():.2%}")
        else:
            st.info("Run optimization to see detailed metrics")

else:
    # Show what will be tested
    st.markdown("## 📝 Configuration Summary")

    st.info(f"""
    **Ready to optimize with:**
    - {len(thresholds)} thresholds from {threshold_min:.2f} to {threshold_max:.2f}
    - {len(selected_models)} embedding models
    - {len(test_df)} test queries
    
    Click "Run Optimization" to begin systematic analysis.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 Tip: Start with a broad threshold range (0.10-0.50), then narrow down based on results</p>
</div>
""", unsafe_allow_html=True)