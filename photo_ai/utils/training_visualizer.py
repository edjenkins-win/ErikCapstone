import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import streamlit as st
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: List[float]
    accuracy: List[float]
    validation_loss: Optional[List[float]] = None
    validation_accuracy: Optional[List[float]] = None
    learning_rate: Optional[List[float]] = None
    timestamps: Optional[List[datetime]] = None

class TrainingVisualizer:
    """Handles visualization of training metrics."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.metrics_history: Dict[str, TrainingMetrics] = {}
        self.current_epoch = 0
    
    def update_metrics(self, 
                      agent_name: str,
                      loss: float,
                      accuracy: float,
                      validation_loss: Optional[float] = None,
                      validation_accuracy: Optional[float] = None,
                      learning_rate: Optional[float] = None) -> None:
        """Update metrics for an agent.
        
        Args:
            agent_name: Name of the agent
            loss: Current loss value
            accuracy: Current accuracy value
            validation_loss: Optional validation loss
            validation_accuracy: Optional validation accuracy
            learning_rate: Optional learning rate
        """
        if agent_name not in self.metrics_history:
            self.metrics_history[agent_name] = TrainingMetrics(
                loss=[], accuracy=[],
                validation_loss=[], validation_accuracy=[],
                learning_rate=[], timestamps=[]
            )
        
        metrics = self.metrics_history[agent_name]
        metrics.loss.append(loss)
        metrics.accuracy.append(accuracy)
        
        if validation_loss is not None:
            metrics.validation_loss.append(validation_loss)
        if validation_accuracy is not None:
            metrics.validation_accuracy.append(validation_accuracy)
        if learning_rate is not None:
            metrics.learning_rate.append(learning_rate)
        
        metrics.timestamps.append(datetime.now())
        self.current_epoch += 1
    
    def plot_metrics(self, agent_name: str) -> None:
        """Plot training metrics for an agent.
        
        Args:
            agent_name: Name of the agent to plot metrics for
        """
        if agent_name not in self.metrics_history:
            st.warning(f"No metrics found for {agent_name}")
            return
        
        metrics = self.metrics_history[agent_name]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot loss
        ax1.plot(metrics.loss, label='Training Loss')
        if metrics.validation_loss:
            ax1.plot(metrics.validation_loss, label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(metrics.accuracy, label='Training Accuracy')
        if metrics.validation_accuracy:
            ax2.plot(metrics.validation_accuracy, label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and display
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display current metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Loss", f"{metrics.loss[-1]:.4f}")
        with col2:
            st.metric("Current Accuracy", f"{metrics.accuracy[-1]:.4f}")
        with col3:
            if metrics.learning_rate:
                st.metric("Learning Rate", f"{metrics.learning_rate[-1]:.6f}")
    
    def plot_learning_rate(self, agent_name: str) -> None:
        """Plot learning rate schedule for an agent.
        
        Args:
            agent_name: Name of the agent to plot learning rate for
        """
        if agent_name not in self.metrics_history or not self.metrics_history[agent_name].learning_rate:
            st.warning(f"No learning rate data found for {agent_name}")
            return
        
        metrics = self.metrics_history[agent_name]
        
        plt.figure(figsize=(10, 4))
        plt.plot(metrics.learning_rate)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        st.pyplot(plt)
    
    def plot_metric_comparison(self, metric: str) -> None:
        """Plot comparison of a specific metric across all agents.
        
        Args:
            metric: Metric to compare ('loss' or 'accuracy')
        """
        plt.figure(figsize=(10, 6))
        
        for agent_name, metrics in self.metrics_history.items():
            if metric == 'loss':
                values = metrics.loss
                if metrics.validation_loss:
                    values = metrics.validation_loss
            else:  # accuracy
                values = metrics.accuracy
                if metrics.validation_accuracy:
                    values = metrics.validation_accuracy
            
            plt.plot(values, label=agent_name)
        
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics_history.clear()
        self.current_epoch = 0 