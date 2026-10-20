import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.ml_engine import FEATURE_COLS, FEATURE_LABELS

def create_empty_figure(message="No data matches the selected filters."):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=380)
    return fig

def create_system_architecture_diagram():
    """
    Renders an interactive Sankey/Flowchart diagram representing the project novel workflow.
    """
    fig = go.Figure()

    stages = [
        "1. Social Media Usage Data",
        "2. Behavioral Feature Extraction",
        "3. Personal Baseline Modeling",
        "4. Behavioral Change Detection (Δ)",
        "5. XGBoost Model",
        "6. Addiction-Risk Score",
        "7. SHAP Explanation + PARI Alert"
    ]
    
    # Process Flow Flowchart Nodes
    x_pos = [0.0, 0.16, 0.33, 0.50, 0.66, 0.83, 1.0]
    y_pos = [0.5, 0.5,  0.5,  0.5,  0.5,  0.5,  0.5]

    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode="markers+text",
        marker=dict(size=42, color="#2563EB", symbol="square", line=dict(color="#1E3A8A", width=2)),
        text=[f"<b>{s}</b>" for s in stages],
        textposition="top center",
        hoverinfo="text",
        hovertext=[
            "Raw 4-week daily digital traces (Screen time, Sessions, Night usage, Checking freq)",
            "Extracted structural metrics: screen time, session freq, night usage, variability",
            "Personalized baseline computed per student (Week 1 / initial usage norms)",
            "Novelty: Δ screen time, Δ night usage, Δ checking frequency relative to personal baseline",
            "Supervised gradient boosted decision tree classifier trained on change trajectories",
            "Continuous model output (0-100%) reflecting trajectory escalation probability",
            "Local/Global SHAP feature attribution + Personalized Addiction Risk Index (PARI)"
        ]
    ))

    # Add connecting arrows
    for i in range(len(x_pos) - 1):
        fig.add_annotation(
            x=x_pos[i+1], y=y_pos[i+1],
            ax=x_pos[i], ay=y_pos[i],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#0D9488"
        )

    fig.update_layout(
        title="<b>System Architecture Flowchart: Personal Temporal Modeling & Explainable AI</b>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.2, 0.8]),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=240
    )
    return fig

def create_student_trajectory_chart(df, student_id=None):
    """
    Plots the 4-week temporal trajectory of a selected student vs their baseline.
    """
    if df.empty:
        return create_empty_figure()
    
    if student_id is None or student_id not in df['Student_ID'].values:
        student_row = df.iloc[0]
    else:
        student_row = df[df['Student_ID'] == student_id].iloc[0]

    weeks = ["Week 1 (Base)", "Week 2", "Week 3", "Week 4 (Current)"]
    screen_vals = [student_row['W1_Screen'], student_row['W2_Screen'], student_row['W3_Screen'], student_row['W4_Screen']]

    fig = go.Figure()

    # Trajectory line
    fig.add_trace(go.Scatter(
        x=weeks, y=screen_vals,
        mode="lines+markers+text",
        name="Daily Screen Time (hrs)",
        text=[f"{v}h" for v in screen_vals],
        textposition="top center",
        line=dict(color="#2563EB", width=4),
        marker=dict(size=10, color="#1E3A8A")
    ))

    # Personal Baseline reference line
    baseline = student_row['Base_Screen_Time']
    fig.add_trace(go.Scatter(
        x=weeks, y=[baseline]*4,
        mode="lines",
        name="Personal Baseline",
        line=dict(color="#EF4444", width=2, dash="dash")
    ))

    title_text = (
        f"<b>Temporal Trajectory: {student_row['Student_ID']}</b> "
        f"({student_row['Gender']}, {student_row['Academic_Level']} - {student_row['Most_Used_Platform']})<br>"
        f"<sup>PARI Score: {student_row['PARI_Score']} | Risk Level: {student_row['PARI_Category']}</sup>"
    )

    fig.update_layout(
        title=title_text,
        xaxis_title="Timeline",
        yaxis_title="Daily Screen Time (Hours)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=65, b=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_pari_category_distribution(df):
    """
    Distribution of Personalized Addiction Risk Index (PARI) across the 4 risk categories.
    """
    if df.empty:
        return create_empty_figure()

    order = [
        "🟢 Stable/Low Risk", 
        "🟡 Increasing Risk", 
        "🟠 Persistent High Risk", 
        "🔴 Rapidly Increasing Risk"
    ]
    color_map = {
        "🟢 Stable/Low Risk": "#10B981",
        "🟡 Increasing Risk": "#F59E0B",
        "🟠 Persistent High Risk": "#F97316",
        "🔴 Rapidly Increasing Risk": "#EF4444"
    }

    cat_counts = df['PARI_Category'].value_counts().reindex(order).fillna(0).reset_index()
    cat_counts.columns = ['Risk_Category', 'Count']

    fig = px.bar(
        cat_counts,
        x='Risk_Category',
        y='Count',
        color='Risk_Category',
        title="<b>Personalized Addiction Risk Index (PARI) Category Distribution</b>",
        color_discrete_map=color_map,
        text_auto=True
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        showlegend=False,
        xaxis_title="PARI Classification",
        yaxis_title="Number of Students"
    )
    return fig

def create_behavioral_change_impact_plot(df):
    """
    Scatter plot showing how Δ Night Usage and Δ Screen Time drive PARI Risk Score.
    """
    if df.empty:
        return create_empty_figure()

    fig = px.scatter(
        df,
        x="Delta_Night_Usage",
        y="PARI_Score",
        color="PARI_Category",
        size="Curr_Screen_Time",
        hover_data=["Student_ID", "Delta_Screen_Time", "Delta_Checking_Freq", "GPA"],
        title="<b>Behavioral Change Impact: Δ Night Usage vs PARI Risk Score</b>",
        color_discrete_map={
            "🟢 Stable/Low Risk": "#10B981",
            "🟡 Increasing Risk": "#F59E0B",
            "🟠 Persistent High Risk": "#F97316",
            "🔴 Rapidly Increasing Risk": "#EF4444"
        }
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        xaxis_title="Δ Night Usage relative to Personal Baseline (Hours)",
        yaxis_title="PARI Risk Score (0 - 100)"
    )
    return fig

def create_global_shap_importance_plot(df):
    """
    Global SHAP Feature Importance bar chart across all students.
    """
    if df.empty:
        return create_empty_figure()

    shap_cols = [f'SHAP_{c}' for c in FEATURE_COLS]
    mean_abs_shap = df[shap_cols].abs().mean().reset_index()
    mean_abs_shap.columns = ['Feature_SHAP', 'Mean_Abs_SHAP']
    
    # Map back to readable feature names
    mean_abs_shap['Feature_Name'] = mean_abs_shap['Feature_SHAP'].apply(
        lambda col: FEATURE_LABELS.get(col.replace('SHAP_', ''), col.replace('SHAP_', ''))
    )
    mean_abs_shap['Is_Delta'] = mean_abs_shap['Feature_Name'].apply(lambda name: 'Δ Behavioral Change' if 'Δ' in name or 'Variability' in name else 'Static Usage')

    mean_abs_shap = mean_abs_shap.sort_values(by='Mean_Abs_SHAP', ascending=True)

    fig = px.bar(
        mean_abs_shap,
        x='Mean_Abs_SHAP',
        y='Feature_Name',
        color='Is_Delta',
        orientation='h',
        title="<b>Global SHAP Explainable AI: Behavioral Change vs Static Metrics</b>",
        color_discrete_map={'Δ Behavioral Change': '#2563EB', 'Static Usage': '#9CA3AF'},
        text_auto='.3f'
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        xaxis_title="Mean |SHAP Value| (Impact on Addiction-Risk Probability)",
        yaxis_title="Extracted Behavioral Feature"
    )
    return fig

def create_local_shap_explanation_plot(df, student_id=None):
    """
    Local SHAP feature contribution bar chart for a selected student.
    """
    if df.empty:
        return create_empty_figure()

    if student_id is None or student_id not in df['Student_ID'].values:
        student_row = df.iloc[0]
    else:
        student_row = df[df['Student_ID'] == student_id].iloc[0]

    shap_data = []
    for col in FEATURE_COLS:
        shap_val = student_row[f'SHAP_{col}']
        feat_name = FEATURE_LABELS.get(col, col)
        raw_val = student_row[col]
        shap_data.append({
            'Feature': feat_name,
            'SHAP_Value': shap_val,
            'Raw_Value': raw_val,
            'Direction': 'Increased Risk' if shap_val >= 0 else 'Reduced Risk'
        })

    shap_df = pd.DataFrame(shap_data).sort_values(by='SHAP_Value', ascending=True)

    fig = px.bar(
        shap_df,
        x='SHAP_Value',
        y='Feature',
        color='Direction',
        orientation='h',
        title=f"<b>Local SHAP Explanation for {student_row['Student_ID']}</b> (PARI = {student_row['PARI_Score']})",
        color_discrete_map={'Increased Risk': '#EF4444', 'Reduced Risk': '#10B981'},
        hover_data=['Raw_Value'],
        text_auto='.3f'
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        xaxis_title="SHAP Value (Contribution to Individual Risk Score)",
        yaxis_title="Feature"
    )
    return fig

def create_platform_pari_chart(df):
    """
    Average PARI score and risk breakdown across social media platforms.
    """
    if df.empty:
        return create_empty_figure()

    platform_summary = df.groupby(['Most_Used_Platform', 'PARI_Category'], observed=False).size().reset_index(name='Count')
    
    fig = px.bar(
        platform_summary,
        x='Most_Used_Platform',
        y='Count',
        color='PARI_Category',
        title="<b>PARI Risk Severity by Most Used Social Media Platform</b>",
        color_discrete_map={
            "🟢 Stable/Low Risk": "#10B981",
            "🟡 Increasing Risk": "#F59E0B",
            "🟠 Persistent High Risk": "#F97316",
            "🔴 Rapidly Increasing Risk": "#EF4444"
        },
        category_orders={
            "PARI_Category": ["🟢 Stable/Low Risk", "🟡 Increasing Risk", "🟠 Persistent High Risk", "🔴 Rapidly Increasing Risk"]
        }
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        xaxis_title="Platform",
        yaxis_title="Number of Students"
    )
    return fig
