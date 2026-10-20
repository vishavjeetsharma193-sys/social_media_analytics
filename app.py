import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import os
import traceback
import webbrowser
import threading

from src.data_processing import generate_temporal_dataset, preprocess_data
from src.ml_engine import train_xgboost_and_compute_pari
from src.visualizations import (
    create_system_architecture_diagram,
    create_student_trajectory_chart,
    create_pari_category_distribution,
    create_behavioral_change_impact_plot,
    create_global_shap_importance_plot,
    create_local_shap_explanation_plot,
    create_platform_pari_chart,
    create_empty_figure
)

# Ingest and Process Temporal Behavioral Dataset
PROCESSED_DATA_PATH = "data/students_behavioral_change_features.csv"
if not os.path.exists(PROCESSED_DATA_PATH):
    generate_temporal_dataset()

df_features = preprocess_data()
df_final = train_xgboost_and_compute_pari(df_features)

# Pre-generate figures
fig_arch = create_system_architecture_diagram()
fig_pari_dist = create_pari_category_distribution(df_final)
fig_change_impact = create_behavioral_change_impact_plot(df_final)
fig_global_shap = create_global_shap_importance_plot(df_final)
fig_platform_pari = create_platform_pari_chart(df_final)

initial_student = df_final['Student_ID'].iloc[0]
fig_initial_trajectory = create_student_trajectory_chart(df_final, initial_student)
fig_initial_local_shap = create_local_shap_explanation_plot(df_final, initial_student)

def open_browser():
    try:
        webbrowser.open_new("http://127.0.0.1:8050/")
    except Exception as e:
        print("Failed to open browser automatically:", e)

# Initialize Dash Application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Personalized Temporal Social Media Addiction Risk Analytics"

# Navigation Header
navbar = dbc.NavbarSimple(
    brand="Personalized Temporal Modeling of Social Media Addiction Risk",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4 shadow-sm fw-bold"
)

# Left Sidebar Control Panel
controls = dbc.Card(
    [
        html.H5("Analytics & Student Filters", className="card-title text-primary fw-bold"),
        html.Hr(),
        html.Div([
            html.Label("Inspect Specific Student:", className="fw-bold text-dark"),
            dcc.Dropdown(
                id="student-dropdown",
                options=[{"label": f"{s} ({df_final[df_final['Student_ID']==s]['PARI_Category'].iloc[0]})", "value": s} for s in df_final["Student_ID"]],
                value=initial_student,
                clearable=False
            ),
        ]),
        html.Div([
            html.Label("Filter Risk Level:", className="fw-bold mt-3 text-dark"),
            dcc.Dropdown(
                id="risk-filter",
                options=[{"label": "All Risk Levels", "value": "All"}] + [{"label": str(r), "value": str(r)} for r in sorted(df_final["PARI_Category"].unique())],
                value="All",
                clearable=False
            ),
        ]),
        html.Div([
            html.Label("Filter Academic Level:", className="fw-bold mt-3 text-dark"),
            dcc.Dropdown(
                id="academic-filter",
                options=[{"label": "All Academic Levels", "value": "All"}] + [{"label": str(a), "value": str(a)} for a in sorted(df_final["Academic_Level"].unique())],
                value="All",
                clearable=False
            ),
        ]),
        html.Div([
            html.Label("Filter Platform:", className="fw-bold mt-3 text-dark"),
            dcc.Dropdown(
                id="platform-filter",
                options=[{"label": "All Platforms", "value": "All"}] + [{"label": str(p), "value": str(p)} for p in sorted(df_final["Most_Used_Platform"].unique())],
                value="All",
                clearable=False
            ),
        ]),
        html.Hr(),
        html.Div([
            html.P("Active Cohort Size:", className="text-muted mb-0 small"),
            html.H4(f"{len(df_final)} / {len(df_final)} Students", id="sample-count", className="text-primary fw-bold mb-0")
        ], className="text-center py-2 bg-light rounded border")
    ],
    body=True,
    className="shadow-sm mb-4"
)

# Alert Banner Container for Selected Student
alert_container = html.Div(id="alert-banner-container", className="mb-3")

# Tab 1: System Architecture & Trajectory Modeling
tab1_content = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_arch), md=12)
        ]),
        alert_container,
        dbc.Row([
            dbc.Col(dcc.Graph(id="student-trajectory-graph", figure=fig_initial_trajectory), md=8),
            dbc.Col(
                dbc.Card(
                    [
                        html.H5("Personal Baseline vs Current Metrics", className="card-title text-primary fw-bold"),
                        html.Div(id="student-metrics-table")
                    ],
                    body=True,
                    className="bg-light border-0 shadow-sm h-100"
                ),
                md=4
            )
        ])
    ]),
    className="mt-2 border-0 shadow-sm"
)

# Tab 2: Dynamic Change & PARI Score Dashboard
tab2_content = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dcc.Graph(id="pari-dist-graph", figure=fig_pari_dist), md=6),
            dbc.Col(dcc.Graph(id="change-impact-graph", figure=fig_change_impact), md=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="platform-pari-graph", figure=fig_platform_pari), md=12, className="mt-3")
        ])
    ]),
    className="mt-2 border-0 shadow-sm"
)

# Tab 3: XGBoost & SHAP Explainable AI (XAI)
tab3_content = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dcc.Graph(id="global-shap-graph", figure=fig_global_shap), md=6),
            dbc.Col(dcc.Graph(id="local-shap-graph", figure=fig_initial_local_shap), md=6)
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    [
                        html.H5("Key Methodological Innovation & XAI Insights", className="card-title text-primary fw-bold"),
                        html.Ul([
                            html.Li([html.B("Novelty - Behavioral Change (Δ): "), "Rather than relying on static single-questionnaire usage metrics, the model evaluates trajectory changes relative to each student's personal baseline (Δ Screen Time, Δ Night Usage, Δ Checking Freq)."]),
                            html.Li([html.B("Personalized Addiction Risk Index (PARI): "), "Formulated as PARI = Current Risk Probability + Behavioral Change Score + Persistence Score across 4 weeks."]),
                            html.Li([html.B("SHAP Feature Importance: "), "Demonstrates empirically that behavioral change indicators (Δ Night Usage, Δ Screen Time) contribute more significantly to predicted risk than static hours alone."]),
                            html.Li([html.B("Dynamic Warning System: "), "Triggers early-warning alerts when positive trajectory shifts exceed personal tolerance baselines."])
                        ])
                    ],
                    body=True,
                    className="mt-3 bg-light border-0"
                ),
                md=12
            )
        ])
    ]),
    className="mt-2 border-0 shadow-sm"
)

tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Architecture & Trajectory Trajectories", tab_id="tab-1"),
        dbc.Tab(tab2_content, label="Behavioral Change & PARI Index", tab_id="tab-2"),
        dbc.Tab(tab3_content, label="XGBoost & SHAP Explainable AI", tab_id="tab-3")
    ],
    id="analytics-tabs",
    active_tab="tab-1"
)

app.layout = dbc.Container([
    navbar,
    dbc.Row([
        dbc.Col(controls, md=3),
        dbc.Col(tabs, md=9)
    ])
], fluid=True)

# Reactive Callbacks for Graph Updates & Filter Synchronizations
@app.callback(
    [
        Output("pari-dist-graph", "figure"),
        Output("change-impact-graph", "figure"),
        Output("platform-pari-graph", "figure"),
        Output("global-shap-graph", "figure"),
        Output("sample-count", "children")
    ],
    [
        Input("risk-filter", "value"),
        Input("academic-filter", "value"),
        Input("platform-filter", "value")
    ]
)
def update_global_charts(selected_risk, selected_academic, selected_platform):
    try:
        filtered_df = df_final.copy()
        
        if selected_risk and selected_risk != "All":
            filtered_df = filtered_df[filtered_df["PARI_Category"] == selected_risk]
        if selected_academic and selected_academic != "All":
            filtered_df = filtered_df[filtered_df["Academic_Level"] == selected_academic]
        if selected_platform and selected_platform != "All":
            filtered_df = filtered_df[filtered_df["Most_Used_Platform"] == selected_platform]
            
        fig_pari = create_pari_category_distribution(filtered_df)
        fig_impact = create_behavioral_change_impact_plot(filtered_df)
        fig_plat = create_platform_pari_chart(filtered_df)
        fig_shap = create_global_shap_importance_plot(filtered_df)
        
        count_str = f"{len(filtered_df)} / {len(df_final)} Students"
        
        return [fig_pari, fig_impact, fig_plat, fig_shap, count_str]
    except Exception as e:
        print("Error in global callback:", str(e))
        traceback.print_exc()
        empty = create_empty_figure(f"Error: {str(e)}")
        return [empty, empty, empty, empty, "0 / 0"]

@app.callback(
    [
        Output("student-trajectory-graph", "figure"),
        Output("local-shap-graph", "figure"),
        Output("alert-banner-container", "children"),
        Output("student-metrics-table", "children")
    ],
    [
        Input("student-dropdown", "value")
    ]
)
def update_student_inspector(selected_student):
    try:
        if not selected_student or selected_student not in df_final['Student_ID'].values:
            selected_student = df_final['Student_ID'].iloc[0]

        row = df_final[df_final['Student_ID'] == selected_student].iloc[0]

        fig_traj = create_student_trajectory_chart(df_final, selected_student)
        fig_local_shap = create_local_shap_explanation_plot(df_final, selected_student)

        # Alert banner rendering
        category = row['PARI_Category']
        alert_msg = row['Early_Warning_Alert']
        
        if "🔴" in category or "🚨" in alert_msg:
            alert_color = "danger"
        elif "🟠" in category or "⚠️" in alert_msg:
            alert_color = "warning"
        elif "🟡" in category:
            alert_color = "info"
        else:
            alert_color = "success"

        alert_banner = dbc.Alert(
            [
                html.H6(f"Status for {selected_student}: {category}", className="alert-heading fw-bold mb-1"),
                html.P(alert_msg, className="mb-0")
            ],
            color=alert_color,
            className="shadow-sm border-0"
        )

        # Comparative metrics table
        metrics_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Baseline (W1)"), html.Th("Current (W4)"), html.Th("Change (Δ)")])),
            html.Tbody([
                html.Tr([
                    html.Td("Screen Time"), 
                    html.Td(f"{row['Base_Screen_Time']} h"), 
                    html.Td(f"{row['Curr_Screen_Time']} h"), 
                    html.Td(f"+{row['Delta_Screen_Time']} h" if row['Delta_Screen_Time'] >= 0 else f"{row['Delta_Screen_Time']} h", className="fw-bold text-danger" if row['Delta_Screen_Time'] > 1.0 else "text-dark")
                ]),
                html.Tr([
                    html.Td("Night Usage"), 
                    html.Td(f"{row['Base_Night_Usage']} h"), 
                    html.Td(f"{row['Curr_Night_Usage']} h"), 
                    html.Td(f"+{row['Delta_Night_Usage']} h" if row['Delta_Night_Usage'] >= 0 else f"{row['Delta_Night_Usage']} h", className="fw-bold text-danger" if row['Delta_Night_Usage'] > 0.5 else "text-dark")
                ]),
                html.Tr([
                    html.Td("Checking Freq"), 
                    html.Td(f"{row['Base_Checking_Freq']}/h"), 
                    html.Td(f"{row['Curr_Checking_Freq']}/h"), 
                    html.Td(f"+{row['Delta_Checking_Freq']}/h" if row['Delta_Checking_Freq'] >= 0 else f"{row['Delta_Checking_Freq']}/h", className="fw-bold text-warning" if row['Delta_Checking_Freq'] > 3.0 else "text-dark")
                ]),
                html.Tr([
                    html.Td("Session Freq"), 
                    html.Td(f"{row['Base_Session_Freq']}"), 
                    html.Td(f"{row['Curr_Session_Freq']}"), 
                    html.Td(f"+{row['Delta_Session_Freq']}" if row['Delta_Session_Freq'] >= 0 else f"{row['Delta_Session_Freq']}")
                ]),
                html.Tr([
                    html.Td("PARI Risk Index"), 
                    html.Td("-"), 
                    html.Td(f"{row['PARI_Score']} / 100", className="fw-bold text-primary"), 
                    html.Td(row['PARI_Category'])
                ]),
            ])
        ], bordered=True, hover=True, size="sm", className="bg-white rounded")

        return [fig_traj, fig_local_shap, alert_banner, metrics_table]
    except Exception as e:
        print("Error in student inspector callback:", str(e))
        traceback.print_exc()
        empty = create_empty_figure(f"Error: {str(e)}")
        return [empty, empty, html.Div(), html.Div()]

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.0, open_browser).start()
    app.run(debug=True, port=8050)
