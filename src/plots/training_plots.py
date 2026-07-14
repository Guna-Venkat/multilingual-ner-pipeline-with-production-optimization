import json
import os
import matplotlib.pyplot as plt
from typing import Optional

def plot_training_metrics(state_file: str, output_dir: str) -> None:
    """
    Parse trainer_state.json and visualize training loss and validation F1 score over steps.
    """
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        history = state.get('log_history', [])
        
        # Extract values for plotting
        train_loss = [x['loss'] for x in history if 'loss' in x]
        train_steps = [x['step'] for x in history if 'loss' in x]
        
        eval_f1 = [x['eval_f1'] for x in history if 'eval_f1' in x]
        eval_steps = [x['step'] for x in history if 'eval_f1' in x]
        
        # Style plots beautifully
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=100)
        
        # Plot Training Loss
        if train_loss:
            axes[0].plot(train_steps, train_loss, label='Training Loss', color='#1f77b4', linewidth=2)
            axes[0].set_title('Training Loss over Steps', fontsize=12, fontweight='bold', pad=10)
            axes[0].set_xlabel('Step', fontsize=10)
            axes[0].set_ylabel('Loss', fontsize=10)
            axes[0].grid(True, linestyle='--', alpha=0.6)
            axes[0].legend(frameon=True)
        
        # Plot Validation F1
        if eval_f1:
            axes[1].plot(eval_steps, eval_f1, label='Eval F1', color='#2ca02c', marker='o', linewidth=2)
            axes[1].set_title('Validation F1 Score', fontsize=12, fontweight='bold', pad=10)
            axes[1].set_xlabel('Step', fontsize=10)
            axes[1].set_ylabel('F1 Score', fontsize=10)
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].legend(frameon=True)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "training_plots.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Training metrics plot saved to {save_path}")
        
    except FileNotFoundError:
        print(f"Warning: The file {state_file} was not found. Skipping plot generation.")
    except Exception as e:
        print(f"Could not generate training plots: {e}")

def plot_knowledge_transfer(state_file: str, output_dir: str) -> None:
    """
    Parse student trainer_state.json to visualize the distillation process.
    """
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        history = state.get('log_history', [])
        
        steps = []
        total_loss = []
        distill_loss = []
        hard_loss = []
        
        eval_steps = []
        eval_f1 = []

        for entry in history:
            if 'loss' in entry:
                steps.append(entry['step'])
                total_loss.append(entry['loss'])
                if 'distillation_loss' in entry:
                    distill_loss.append(entry['distillation_loss'])
                if 'hard_label_loss' in entry:
                    hard_loss.append(entry['hard_label_loss'])
            
            if 'eval_f1' in entry:
                eval_steps.append(entry['step'])
                eval_f1.append(entry['eval_f1'])

        # Style plots beautifully
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
        
        # Plot 1: Total Training Loss
        if total_loss:
            axes[0].plot(steps, total_loss, label='Total Loss', color='#1f77b4', linewidth=2)
            axes[0].set_title('Student Training Loss', fontsize=12, fontweight='bold', pad=10)
            axes[0].set_xlabel('Step', fontsize=10)
            axes[0].set_ylabel('Loss', fontsize=10)
            axes[0].grid(True, linestyle='--', alpha=0.6)
            
            if distill_loss:
                axes[0].plot(steps, distill_loss, label='Distill Component', color='#ff7f0e', linestyle='--', alpha=0.8)
            if hard_loss:
                axes[0].plot(steps, hard_loss, label='Hard Label Component', color='#d62728', linestyle=':', alpha=0.8)
            axes[0].legend(frameon=True)
        
        # Plot 2: Validation F1 Score
        if eval_f1:
            axes[1].plot(eval_steps, eval_f1, marker='o', color='#2ca02c', linewidth=2, label='Student F1')
            axes[1].set_title('Student Validation F1 Score', fontsize=12, fontweight='bold', pad=10)
            axes[1].set_xlabel('Step', fontsize=10)
            axes[1].set_ylabel('F1 Score', fontsize=10)
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].legend(frameon=True)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "distillation_plots.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Distillation plots saved to {save_path}")
        
    except FileNotFoundError:
        print(f"Warning: The file {state_file} was not found. Skipping distillation plot generation.")
    except Exception as e:
        print(f"Could not generate distillation plots: {e}")
