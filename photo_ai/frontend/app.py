import streamlit as st
import os
import cv2
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, Any, List, Union, Tuple
import json
import tkinter as tk
from tkinter import filedialog
import psutil
import torch
import logging
from photo_ai.utils.performance import ProcessingMode
from photo_ai.core.image_processor import ImageProcessor

# Define supported image formats for file uploaders
SUPPORTED_IMAGE_FORMATS = [
    # Standard formats
    "jpg", "jpeg", "png", "tiff", "bmp", "gif",
    # Modern formats
    "webp", "heif", "heic",
    # Raw formats (may require additional libraries)
    "dng", "arw", "cr2", "nef", "orf", "rw2"
]

from photo_ai.agents.batch_agent import BatchAgent
from photo_ai.agents.color_agent import ColorAgent
from photo_ai.agents.exposure_agent import ExposureAgent
from photo_ai.agents.skin_agent import SkinAgent
from photo_ai.agents.composition_agent import CompositionAgent
from photo_ai.agents.background_agent import BackgroundAgent
from photo_ai.agents.noise_agent import NoiseAgent
from photo_ai.agents.style_agent import StyleAgent
from photo_ai.agents.training_agent import TrainingAgent
from photo_ai.storage.model_storage import ModelStorage

# Configure logging to suppress GPU info logs
logging.getLogger("torch.backends.mps").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PerformanceOptimizer").setLevel(logging.WARNING)  # Suppress PerformanceOptimizer info logs

# Initialize agents
batch_agent = BatchAgent()
color_agent = ColorAgent()
exposure_agent = ExposureAgent()
skin_agent = SkinAgent()
composition_agent = CompositionAgent()
background_agent = BackgroundAgent()
noise_agent = NoiseAgent()
style_agent = StyleAgent()

# Agent processing order
PROCESSING_ORDER = [
    composition_agent,
    noise_agent,
    exposure_agent,
    color_agent,
    skin_agent,
    background_agent
]

def process_single_image(image: np.ndarray) -> np.ndarray:
    """Process a single image using all agents."""
    processed = image.copy()
    for agent in PROCESSING_ORDER:
        processed = agent.process(processed)
    return processed

def create_config_widget(agent_name: str, key: str, value: Any) -> Any:
    """Create appropriate widget based on value type with unique key."""
    widget_key = f"{agent_name}_{key}"

    if isinstance(value, bool):
        return st.checkbox(key, value=value, key=widget_key)
    elif isinstance(value, int):
        return st.slider(
            key,
            min_value=0,
            max_value=100,
            value=value,
            step=1,
            key=widget_key
        )
    elif isinstance(value, float):
        return st.slider(
            key,
            min_value=0.0,
            max_value=1.0,
            value=value,
            step=0.01,
            key=widget_key
        )
    elif isinstance(value, list):
        if all(isinstance(x, str) for x in value):
            return st.multiselect(
                key,
                options=value,
                default=value,
                key=widget_key
            )
        elif all(isinstance(x, (int, float)) for x in value):
            return st.multiselect(
                key,
                options=value,
                default=value,
                key=widget_key
            )
    return st.text_input(key, value=str(value), key=widget_key)

def select_directory(title: str) -> str:
    """Open a directory selection dialog."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    directory = filedialog.askdirectory(title=title)
    root.destroy()
    return directory

# Using ImageProcessor.load_image instead of a custom function

def show_home():
    """Show the home page."""
    st.title("AI Photo Editor")
    st.markdown("""
    Welcome to the AI Photo Editor! This application uses AI to enhance your photos.

    Features:
    - Single image processing
    - Batch processing
    - Model training
    - Model management
    """)

    # Single image processing
    st.header("Single Image Processing")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=SUPPORTED_IMAGE_FORMATS
    )

    if uploaded_file is not None:
        # Read image using ImageProcessor
        image = ImageProcessor.load_image(uploaded_file)

        # Display original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Process image
        if st.button("Process Image"):
            with st.spinner("Processing image..."):
                processed = process_single_image(image)

                # Display processed image
                st.subheader("Processed Image")
                st.image(processed, use_column_width=True)

                # Download button using ImageProcessor
                processed_bytes = ImageProcessor.image_to_bytes(processed)
                st.download_button(
                    label="Download Processed Image",
                    data=processed_bytes,
                    file_name="processed_" + uploaded_file.name,
                    mime="image/jpeg"
                )

def show_style_transfer():
    """Show the style transfer interface."""
    st.title("Style Transfer")

    # Initialize style agent if not already done
    if 'style_agent' not in st.session_state:
        st.session_state.style_agent = StyleAgent()

    # Create columns for image uploads
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Content Image")
        content_file = st.file_uploader(
            "Upload content image",
            type=SUPPORTED_IMAGE_FORMATS,
            key="content_image"
        )

    with col2:
        st.subheader("Style Image")
        style_file = st.file_uploader(
            "Upload style image",
            type=SUPPORTED_IMAGE_FORMATS,
            key="style_image"
        )

    # Display uploaded images
    if content_file and style_file:
        col1, col2 = st.columns(2)
        with col1:
            content_image = ImageProcessor.load_image(content_file)
            st.image(content_image, use_column_width=True, caption="Content Image")

        with col2:
            style_image = ImageProcessor.load_image(style_file)
            st.image(style_image, use_column_width=True, caption="Style Image")

        # Style transfer settings
        st.subheader("Style Transfer Settings")
        col1, col2, col3 = st.columns(3)

        with col1:
            style_weight = st.slider(
                "Style Weight",
                min_value=1e4,
                max_value=1e7,
                value=1e6,
                step=1e4,
                format="%.0e"
            )

        with col2:
            content_weight = st.slider(
                "Content Weight",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

        with col3:
            num_steps = st.slider(
                "Number of Steps",
                min_value=100,
                max_value=1000,
                value=300,
                step=50
            )

        # Update agent configuration
        st.session_state.style_agent.config.update({
            'style_weight': style_weight,
            'content_weight': content_weight,
            'num_steps': num_steps
        })

        # Process button
        if st.button("Apply Style Transfer"):
            with st.spinner("Applying style transfer..."):
                # Process images
                result_image, metrics = st.session_state.style_agent.process(
                    content_image,
                    style_image,
                    num_steps
                )

                # Display result
                st.subheader("Result")
                st.image(result_image, use_column_width=True, caption="Stylized Image")

                # Display metrics
                st.subheader("Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Loss", f"{metrics['final_loss']:.4f}")
                with col2:
                    st.metric("Style Loss", f"{metrics['style_loss']:.4f}")
                with col3:
                    st.metric("Content Loss", f"{metrics['content_loss']:.4f}")

                # Download button using ImageProcessor
                result_bytes = ImageProcessor.image_to_bytes(result_image)
                st.download_button(
                    label="Download Result",
                    data=result_bytes,
                    file_name="stylized_" + content_file.name,
                    mime="image/jpeg"
                )

                # Training visualization
                st.subheader("Training Progress")
                st.session_state.style_agent.visualizer.plot_metrics(
                    st.session_state.style_agent.__class__.__name__
                )

def show_training():
    """Show the training interface."""
    st.title("Model Training")

    # Training mode selection
    training_mode = st.radio(
        "Training Mode",
        ["Single Pair", "Directory Pairs"],
        key="training_mode"
    )

    if training_mode == "Single Pair":
        # Single pair training
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Before Image")
            before_file = st.file_uploader(
                "Upload before image",
                type=SUPPORTED_IMAGE_FORMATS,
                key="before_image"
            )

        with col2:
            st.subheader("After Image")
            after_file = st.file_uploader(
                "Upload after image",
                type=SUPPORTED_IMAGE_FORMATS,
                key="after_image"
            )

        if before_file and after_file:
            col1, col2 = st.columns(2)
            with col1:
                before_image = ImageProcessor.load_image(before_file)
                st.image(before_image, use_column_width=True, caption="Before")

            with col2:
                after_image = ImageProcessor.load_image(after_file)
                st.image(after_image, use_column_width=True, caption="After")

            # Training options
            st.subheader("Training Options")
            col1, col2 = st.columns(2)

            with col1:
                learning_rate = st.slider(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.001,
                    step=0.0001,
                    format="%.4f"
                )

            with col2:
                epochs = st.slider(
                    "Number of Epochs",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1
                )

            # Agent selection
            st.subheader("Select Agents to Train")
            selected_agents = st.multiselect(
                "Agents",
                options=[agent.__class__.__name__ for agent in PROCESSING_ORDER],
                default=[agent.__class__.__name__ for agent in PROCESSING_ORDER]
            )

            if st.button("Start Training"):
                with st.spinner("Training models..."):
                    # Train selected agents
                    for agent in PROCESSING_ORDER:
                        if agent.__class__.__name__ in selected_agents:
                            agent.learn(before_image, after_image)

                    st.success("Training complete!")

                    # Show training results
                    st.subheader("Training Results")
                    for agent in PROCESSING_ORDER:
                        if agent.__class__.__name__ in selected_agents:
                            status = agent.get_status()
                            st.markdown(f"**{status['name']}**")

                            if 'visualizer' in status:
                                status['visualizer'].plot_metrics(agent.__class__.__name__)
                            else:
                                st.json(status)

    else:  # Directory Pairs
        st.subheader("Directory Training")

        col1, col2 = st.columns(2)

        with col1:
            before_dir = st.text_input(
                "Before Images Directory",
                value=st.session_state.get("before_dir", "")
            )

        with col2:
            after_dir = st.text_input(
                "After Images Directory",
                value=st.session_state.get("after_dir", "")
            )

        # Directory validation
        if before_dir and not os.path.isdir(before_dir):
            st.error(f"Before directory does not exist: {before_dir}")
        if after_dir and not os.path.isdir(after_dir):
            st.error(f"After directory does not exist: {after_dir}")

        if before_dir and after_dir and os.path.isdir(before_dir) and os.path.isdir(after_dir):
            # Training options
            st.subheader("Training Options")
            col1, col2 = st.columns(2)

            with col1:
                learning_rate = st.slider(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.001,
                    step=0.0001,
                    format="%.4f"
                )

            with col2:
                epochs = st.slider(
                    "Number of Epochs",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1
                )

            # Agent selection
            st.subheader("Select Agents to Train")
            selected_agents = st.multiselect(
                "Agents",
                options=[agent.__class__.__name__ for agent in PROCESSING_ORDER],
                default=[agent.__class__.__name__ for agent in PROCESSING_ORDER]
            )

            if st.button("Start Training"):
                with st.spinner("Training models..."):
                    # Train selected agents
                    for agent in PROCESSING_ORDER:
                        if agent.__class__.__name__ in selected_agents:
                            agent.learn_directory(before_dir, after_dir)

                    st.success("Training complete!")

                    # Show training results
                    st.subheader("Training Results")
                    for agent in PROCESSING_ORDER:
                        if agent.__class__.__name__ in selected_agents:
                            status = agent.get_status()
                            st.markdown(f"**{status['name']}**")

                            if 'visualizer' in status:
                                status['visualizer'].plot_metrics(agent.__class__.__name__)
                            else:
                                st.json(status)

def show_batch_processing():
    """Show the batch processing page."""
    st.header("Batch Processing")

    # Directory selection
    st.markdown("### Select Directories")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Input Directory")
        input_dir = st.text_input(
            "Path to input directory",
            value=st.session_state.get("input_dir", "")
        )
        if st.button("📁 Browse Input Directory"):
            st.info("Please enter the full path to your input directory in the text field above")

    with col2:
        st.markdown("#### Output Directory")
        output_dir = st.text_input(
            "Path to output directory",
            value=st.session_state.get("output_dir", "")
        )
        if st.button("📁 Browse Output Directory"):
            st.info("Please enter the full path to your output directory in the text field above")

    # Directory validation
    if input_dir and not os.path.isdir(input_dir):
        st.error(f"Input directory does not exist: {input_dir}")
    if output_dir and not os.path.isdir(output_dir):
        st.warning(f"Output directory does not exist. It will be created: {output_dir}")

    # Processing options
    st.markdown("### Processing Options")
    col1, col2 = st.columns(2)
    with col1:
        purge_low_quality = st.checkbox("Purge Low Quality Photos", value=True)
    with col2:
        min_rating = st.slider("Minimum Rating to Keep", 1, 5, 2, step=1)

    if st.button("Process Directory"):
        if not input_dir or not output_dir:
            st.error("Please provide both input and output directory paths")
        else:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            with st.spinner("Processing directory..."):
                # Update batch agent config
                batch_agent.config['min_rating'] = min_rating

                # Process directory
                results = batch_agent.process_directory(
                    input_dir,
                    output_dir,
                    PROCESSING_ORDER,
                    purge_low_quality
                )

                # Display results
                st.success(f"""
                Processing complete!
                - Total photos: {results['total_photos']}
                - Processed photos: {results['processed_photos']}
                - Purged photos: {results['purged_photos']}
                """)

                if results['errors']:
                    st.warning(f"Encountered {len(results['errors'])} errors during processing")
                    for error in results['errors']:
                        st.error(f"{error['file']}: {error['error']}")

def show_settings():
    """Show the settings page."""
    st.title("Settings")

    # Performance Settings
    st.header("Performance Settings")

    col1, col2 = st.columns(2)

    with col1:
        max_workers = st.slider(
            "Max Workers",
            min_value=1,
            max_value=psutil.cpu_count(),
            value=psutil.cpu_count(),
            help="Maximum number of parallel workers for processing"
        )

        cache_size = st.slider(
            "Cache Size",
            min_value=10,
            max_value=1000,
            value=100,
            help="Number of images to cache in memory"
        )

        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=128,
            value=32,
            help="Number of images to process in each batch"
        )

    with col2:
        # GPU Configuration
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]

            st.subheader("GPU Configuration")
            st.write(f"Available GPUs: {gpu_count}")
            for i, name in enumerate(gpu_names):
                st.write(f"GPU {i}: {name}")

            processing_mode = st.selectbox(
                "Processing Mode",
                options=[mode.value for mode in ProcessingMode],
                format_func=lambda x: x.capitalize(),
                help="Select the processing mode for image operations"
            )
        else:
            st.warning("No GPU available. Using CPU-only mode.")
            processing_mode = ProcessingMode.CPU.value

    # Performance Monitoring
    st.header("Performance Monitoring")

    if st.button("Refresh Metrics"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("CPU Usage", f"{psutil.cpu_percent()}%")

        with col2:
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent}%")

        with col3:
            disk = psutil.disk_usage('/')
            st.metric("Disk Usage", f"{disk.percent}%")

        with col4:
            st.metric("Active Threads", f"{psutil.Process().num_threads()}")

        if gpu_available:
            st.subheader("GPU Metrics")
            gpu_cols = st.columns(gpu_count)
            for i, col in enumerate(gpu_cols):
                with col:
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                    memory_cached = torch.cuda.memory_reserved(i) / 1024**2
                    st.metric(f"GPU {i} Memory", f"{memory_allocated:.1f} MB")
                    st.metric(f"GPU {i} Cache", f"{memory_cached:.1f} MB")

    # Update batch agent configuration
    st.session_state.batch_agent.config.update({
        'max_workers': max_workers,
        'cache_size': cache_size,
        'batch_size': batch_size,
        'processing_mode': processing_mode
    })

    # Agent Settings
    st.markdown("### Agent Settings")

    # Agent configuration
    for agent in PROCESSING_ORDER:
        agent_name = agent.__class__.__name__
        with st.expander(f"{agent_name} Settings"):
            for key, value in agent.config.items():
                new_value = create_config_widget(agent_name, key, value)
                agent.config[key] = new_value

    # Save settings
    if st.button("Save Settings", key="save_settings_button"):
        settings = {
            agent.__class__.__name__: agent.config
            for agent in PROCESSING_ORDER
        }
        st.download_button(
            label="Download Settings",
            data=json.dumps(settings, indent=2),
            file_name="agent_settings.json",
            mime="application/json",
            key="download_settings_button"
        )

def show_model_management():
    """Show the model management interface."""
    st.title("Model Management")

    # Initialize model storage
    model_storage = ModelStorage()

    # Create tabs for different model management functions
    tab1, tab2 = st.tabs(["Save Models", "Load Models"])

    with tab1:
        st.header("Save Current Models")

        # Model description
        description = st.text_area(
            "Model Description",
            help="Provide a description of the current model state"
        )

        # Version (optional)
        version = st.text_input(
            "Version (optional)",
            help="Leave blank for automatic versioning"
        )

        if st.button("Save All Models"):
            if not description:
                st.error("Please provide a description")
            else:
                # Save each agent's model
                saved_models = []
                for agent in PROCESSING_ORDER:
                    try:
                        model_id = agent.save_model(
                            description=description,
                            version=version if version else None
                        )
                        saved_models.append((agent.__class__.__name__, model_id))
                    except Exception as e:
                        st.error(f"Error saving {agent.__class__.__name__}: {str(e)}")

                if saved_models:
                    st.success("Models saved successfully!")
                    for agent_name, model_id in saved_models:
                        st.write(f"{agent_name}: {model_id}")

    with tab2:
        st.header("Load Models")

        # Get all models grouped by agent
        all_models = model_storage.list_models()
        agent_models = {}

        for model in all_models:
            agent_name = model["agent_name"]
            if agent_name not in agent_models:
                agent_models[agent_name] = []
            agent_models[agent_name].append(model)

        # Display models by agent
        for agent_name, models in agent_models.items():
            with st.expander(f"{agent_name} Models"):
                for model in models:
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.write(f"**ID:** {model['model_id']}")
                        st.write(f"**Description:** {model['description']}")
                        st.write(f"**Version:** {model['version']}")
                        st.write(f"**Created:** {model['created_at']}")

                        if model.get("metrics"):
                            st.write("**Metrics:**")
                            for metric, value in model["metrics"].items():
                                st.write(f"- {metric}: {value:.4f}")

                    with col2:
                        if st.button("Load", key=f"load_{model['model_id']}"):
                            try:
                                # Find the corresponding agent
                                agent = next(
                                    (a for a in PROCESSING_ORDER 
                                     if a.__class__.__name__ == agent_name),
                                    None
                                )
                                if agent:
                                    agent.load_model(model['model_id'])
                                    st.success(f"Model {model['model_id']} loaded!")
                                else:
                                    st.error(f"Agent {agent_name} not found")
                            except Exception as e:
                                st.error(f"Error loading model: {str(e)}")

                    with col3:
                        if st.button("Delete", key=f"delete_{model['model_id']}"):
                            try:
                                model_storage.delete_model(model['model_id'])
                                st.success(f"Model {model['model_id']} deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting model: {str(e)}")

                    st.divider()

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state variables if they don't exist
    if 'batch_agent' not in st.session_state:
        st.session_state.batch_agent = BatchAgent()
    if 'style_agent' not in st.session_state:
        st.session_state.style_agent = StyleAgent()
    if 'training_agent' not in st.session_state:
        st.session_state.training_agent = TrainingAgent()

    st.set_page_config(
        page_title="Photo AI",
        page_icon="🖼️",
        layout="wide"
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")

    # Create static navigation links using radio buttons
    page = st.sidebar.radio(
        "",  # Empty label since we have the title above
        ["Home", "Style Transfer", "Training", "Batch Processing", "Settings", "Model Management"],
        label_visibility="collapsed"  # Hide the empty label
    )

    if page == "Home":
        show_home()
    elif page == "Style Transfer":
        show_style_transfer()
    elif page == "Training":
        show_training()
    elif page == "Batch Processing":
        show_batch_processing()
    elif page == "Settings":
        show_settings()
    elif page == "Model Management":
        show_model_management()

if __name__ == "__main__":
    main() 
