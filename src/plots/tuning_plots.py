import os
import pandas as pd
import numpy as np
import optuna
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from optuna.trial import TrialState
from typing import Optional, List, Any

# Set a clean default template for Plotly visualizations
pio.templates.default = "plotly_white"

def plot_optimization_history(study: optuna.study.Study, output_dir: Optional[str] = None) -> go.Figure:
    """
    Generate and show optimization history plot.
    """
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(
        title="Optimization History",
        xaxis_title="Trial",
        yaxis_title="F1 Score",
        width=900,
        height=500
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save as HTML to retain interactivity
        fig.write_html(os.path.join(output_dir, "optimization_history.html"))
        try:
            fig.write_image(os.path.join(output_dir, "optimization_history.png"))
        except Exception:
            pass # Skip if kaleido is not working/installed
            
    fig.show(renderer="iframe")
    return fig

def plot_param_importances(study: optuna.study.Study, output_dir: Optional[str] = None) -> Optional[go.Figure]:
    """
    Generate and show parameter importance plot.
    """
    try:
        evaluator = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
        fig = optuna.visualization.plot_param_importances(study, evaluator=evaluator)
        fig.update_layout(
            title="Parameter Importances (Mean Decrease Impurity)",
            width=800,
            height=500
        )
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.write_html(os.path.join(output_dir, "param_importances.html"))
            try:
                fig.write_image(os.path.join(output_dir, "param_importances.png"))
            except Exception:
                pass
                
        fig.show(renderer="iframe")
        return fig
    except Exception as e:
        print(f"Could not plot importance: {e}")
        return None

def plot_parallel_coordinate(study: optuna.study.Study, output_dir: Optional[str] = None) -> go.Figure:
    """
    Generate and show parallel coordinate plot.
    """
    existing_params = list(study.best_params.keys())
    fig = optuna.visualization.plot_parallel_coordinate(study, params=existing_params)
    fig.update_layout(
        title="Hyperparameter Interactions (Parallel Coordinate)",
        width=1100,
        height=600
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(os.path.join(output_dir, "parallel_coordinate.html"))
        try:
            fig.write_image(os.path.join(output_dir, "parallel_coordinate.png"))
        except Exception:
            pass
            
    fig.show(renderer="iframe")
    return fig

def plot_slice_plot(study: optuna.study.Study, output_dir: Optional[str] = None) -> go.Figure:
    """
    Generate and show parameter slice plot.
    """
    existing_params = list(study.best_params.keys())
    fig = optuna.visualization.plot_slice(study, params=existing_params)
    fig.update_layout(
        title="Slice Plot (Parameter Relationships with F1)",
        width=1200,
        height=600
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(os.path.join(output_dir, "slice_plot.html"))
        try:
            fig.write_image(os.path.join(output_dir, "slice_plot.png"))
        except Exception:
            pass
            
    fig.show(renderer="iframe")
    return fig

def create_interactive_dashboard(study: optuna.study.Study, output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Create a scatter matrix dashboard and return study trials as a DataFrame.
    """
    trials = study.trials
    all_params = set()
    for t in trials:
        if t.state == TrialState.COMPLETE:
            all_params.update(t.params.keys())
    param_names = list(all_params)
    
    data = []
    for trial in trials:
        if trial.state == TrialState.COMPLETE:
            score = trial.value if trial.value is not None else 0.0
            row = {"trial": trial.number, "f1_score": score}
            for param in param_names:
                row[param] = trial.params.get(param, None)
            data.append(row)
            
    if not data:
        print("No completed trials found to plot.")
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    
    fig = px.scatter_matrix(
        df,
        dimensions=param_names,
        color="f1_score",
        symbol="f1_score",
        title="Hyperparameter Relationships (Scatter Matrix)",
        labels={col: col.replace('_', ' ') for col in df.columns},
        width=1200,
        height=800,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=5))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(os.path.join(output_dir, "3d_scatter.html"))
        try:
            fig.write_image(os.path.join(output_dir, "3d_scatter.png"))
        except Exception:
            pass
            
    fig.show(renderer="iframe")
    return df

def save_interactive_visualizations(study: optuna.study.Study, df_trials: pd.DataFrame, output_dir: str) -> None:
    """
    Save all interactive plotly plots as HTML files in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Optimization History
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, "optimization_history.html"))
    except Exception as e:
        print(f"Error saving optimization history: {e}")
        
    # 2. Parameter Importances
    try:
        evaluator = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
        fig = optuna.visualization.plot_param_importances(study, evaluator=evaluator)
        fig.write_html(os.path.join(output_dir, "param_importances.html"))
    except Exception as e:
        print(f"Error saving param importances: {e}")
        
    # 3. Parallel Coordinate Plot
    try:
        existing_params = list(study.best_params.keys())
        fig = optuna.visualization.plot_parallel_coordinate(study, params=existing_params)
        fig.write_html(os.path.join(output_dir, "parallel_coordinate.html"))
    except Exception as e:
        print(f"Error saving parallel coordinate plot: {e}")
        
    # 4. Slice Plot
    try:
        existing_params = list(study.best_params.keys())
        fig = optuna.visualization.plot_slice(study, params=existing_params)
        fig.write_html(os.path.join(output_dir, "slice_plot.html"))
    except Exception as e:
        print(f"Error saving slice plot: {e}")
        
    # 5. Scatter Matrix
    try:
        all_params = set(study.best_params.keys())
        param_names = list(all_params)
        fig = px.scatter_matrix(
            df_trials,
            dimensions=param_names,
            color="f1_score",
            title="Hyperparameter Relationships (Scatter Matrix)",
            width=1200,
            height=800
        )
        fig.write_html(os.path.join(output_dir, "3d_scatter.html"))
    except Exception as e:
        print(f"Error saving scatter matrix: {e}")
        
    print(f"All interactive plots saved to {output_dir}/")
