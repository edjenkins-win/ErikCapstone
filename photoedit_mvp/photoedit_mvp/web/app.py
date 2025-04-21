"""Streamlit web app for the Photo Editor application."""

import streamlit as st
import numpy as np
import io
from PIL import Image
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from photoedit_mvp import analyze_image, apply_adjustments
from photoedit_mvp.utils import load_image, save_image
from photoedit_mvp.analyzer import Adjustment
from photoedit_mvp.styles import get_available_styles, get_style_description

# Import Gen AI modules
from photoedit_mvp.ai_analyzer import AIImageAnalyzer
from photoedit_mvp.nl_processor import NLProcessor
from photoedit_mvp.rag_style_engine import RAGStyleEngine

def main():
    """Main function for the Streamlit web app."""
    st.set_page_config(page_title="Photo Editor", page_icon="📷", layout="wide")
    
    # Add title and description
    st.title("AI Photo Editor")
    st.markdown("""
    Upload an image to analyze and enhance it automatically. 
    See the original and edited images side by side in real-time.
    """)
    
    # Create a sidebar for controls
    st.sidebar.title("Controls")
    
    # Initialize Gen AI components if not already in session state
    if 'ai_analyzer' not in st.session_state:
        st.session_state.ai_analyzer = AIImageAnalyzer()
    
    if 'nl_processor' not in st.session_state:
        # Create NL processor
        nl_processor = NLProcessor()
        
        # Register functions for the NL processor
        from photoedit_mvp.executor import ImageExecutor
        executor = ImageExecutor()
        
        # Register exposure adjustment
        nl_processor.register_function(
            name="adjust_exposure",
            func=executor._apply_exposure,
            description="Adjust the brightness/exposure of the image",
            parameters={
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Amount to adjust exposure (-2.0 to 2.0)"
                    }
                },
                "required": ["amount"]
            }
        )
        
        # Register contrast adjustment
        nl_processor.register_function(
            name="adjust_contrast",
            func=executor._apply_contrast,
            description="Adjust the contrast of the image",
            parameters={
                "type": "object",
                "properties": {
                    "multiplier": {
                        "type": "number",
                        "description": "Contrast multiplier (0.5 to 2.0)"
                    }
                },
                "required": ["multiplier"]
            }
        )
        
        # Register saturation adjustment
        nl_processor.register_function(
            name="adjust_saturation",
            func=executor._apply_saturation,
            description="Adjust the saturation/color intensity of the image",
            parameters={
                "type": "object",
                "properties": {
                    "adjustment": {
                        "type": "number",
                        "description": "Saturation adjustment (-1.0 to 1.0)"
                    }
                },
                "required": ["adjustment"]
            }
        )
        
        # Register temperature adjustment
        nl_processor.register_function(
            name="adjust_temperature",
            func=executor._apply_temperature,
            description="Adjust the color temperature (warmth/coolness) of the image",
            parameters={
                "type": "object",
                "properties": {
                    "adjustment": {
                        "type": "number",
                        "description": "Temperature adjustment (-1.0 to 1.0, negative for cooler, positive for warmer)"
                    }
                },
                "required": ["adjustment"]
            }
        )
        
        # Register sharpness adjustment
        nl_processor.register_function(
            name="adjust_sharpness",
            func=executor._apply_sharpening,
            description="Adjust the sharpness of the image",
            parameters={
                "type": "object",
                "properties": {
                    "strength": {
                        "type": "number",
                        "description": "Sharpening strength (0.0 to 1.0)"
                    }
                },
                "required": ["strength"]
            }
        )
        
        # Register noise reduction
        nl_processor.register_function(
            name="reduce_noise",
            func=executor._apply_noise_reduction,
            description="Reduce noise/grain in the image",
            parameters={
                "type": "object",
                "properties": {
                    "strength": {
                        "type": "number",
                        "description": "Noise reduction strength (0.0 to 1.0)"
                    }
                },
                "required": ["strength"]
            }
        )
        
        # Register style application
        def apply_style(image, style_name):
            return executor.apply(image, [], style_name)
            
        nl_processor.register_function(
            name="apply_style",
            func=apply_style,
            description="Apply a predefined style to the image",
            parameters={
                "type": "object",
                "properties": {
                    "style_name": {
                        "type": "string",
                        "description": "Name of the style to apply"
                    }
                },
                "required": ["style_name"]
            }
        )
        
        st.session_state.nl_processor = nl_processor
    
    if 'rag_style_engine' not in st.session_state:
        st.session_state.rag_style_engine = RAGStyleEngine()
    
    # Add file uploader to sidebar
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tiff"])
    
    if uploaded_file is not None:
        # Load the image
        image_bytes = uploaded_file.getvalue()
        image = load_image(io.BytesIO(image_bytes))
        
        # Store the original image in session state
        if 'original_image' not in st.session_state:
            st.session_state.original_image = image.copy()
            
            # Analyze the image
            st.session_state.adjustments = analyze_image(image)
            
            # Initialize edited image
            st.session_state.edited_image = image.copy()
            
            # Clear AI analysis results
            if 'ai_analysis' in st.session_state:
                del st.session_state.ai_analysis
            
            # Clear AI style recommendations
            if 'style_recommendations' in st.session_state:
                del st.session_state.style_recommendations
        
        # Create columns for the images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(st.session_state.original_image, use_container_width=True)
        
        with col2:
            st.subheader("Edited Image")
            st.image(st.session_state.edited_image, use_container_width=True)
        
        # Create edit mode tabs
        tabs = st.tabs(["Auto Adjustments", "Manual Adjustments", "Style Presets", "AI Assistant"])
        
        with tabs[0]:  # Auto Adjustments tab
            st.subheader("Recommended Adjustments")
            
            # Display recommended adjustments
            if st.session_state.adjustments:
                # Create a checkbox for each adjustment
                selected_adjustments = []
                for i, adj in enumerate(st.session_state.adjustments):
                    if st.checkbox(
                        f"{adj.description} ({adj.suggested} {adj.unit})", 
                        value=True,
                        key=f"adj_{i}"
                    ):
                        selected_adjustments.append(adj)
                
                # Apply button
                if st.button("Apply Selected Adjustments"):
                    # Apply the selected adjustments
                    result = apply_adjustments(st.session_state.original_image, selected_adjustments)
                    st.session_state.edited_image = result
                    st.rerun()
            else:
                st.info("No adjustments recommended for this image.")
        
        with tabs[1]:  # Manual Adjustments tab
            st.subheader("Manual Adjustments")
            
            # Create sliders for manual adjustments
            exposure = st.slider("Exposure", -2.0, 2.0, 0.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
            saturation = st.slider("Saturation", -1.0, 1.0, 0.0, 0.1)
            sharpness = st.slider("Sharpness", 0.0, 1.0, 0.0, 0.05)
            temperature = st.slider("Temperature", -1.0, 1.0, 0.0, 0.1)
            noise_reduction = st.slider("Noise Reduction", 0.0, 1.0, 0.0, 0.05)
            
            # Create manual adjustments list
            manual_adjustments = []
            
            if exposure != 0.0:
                manual_adjustments.append(Adjustment(
                    parameter="exposure",
                    suggested=exposure,
                    unit="EV",
                    description="Adjust exposure"
                ))
            
            if contrast != 1.0:
                manual_adjustments.append(Adjustment(
                    parameter="contrast",
                    suggested=contrast,
                    unit="multiplier",
                    description="Adjust contrast"
                ))
            
            if saturation != 0.0:
                manual_adjustments.append(Adjustment(
                    parameter="saturation",
                    suggested=saturation,
                    unit="shift",
                    description="Adjust saturation"
                ))
            
            if sharpness > 0.0:
                manual_adjustments.append(Adjustment(
                    parameter="sharpening",
                    suggested=sharpness,
                    unit="strength",
                    description="Apply sharpening"
                ))
            
            if temperature != 0.0:
                manual_adjustments.append(Adjustment(
                    parameter="temperature",
                    suggested=temperature,
                    unit="shift",
                    description="Adjust temperature"
                ))
            
            if noise_reduction > 0.0:
                manual_adjustments.append(Adjustment(
                    parameter="noise_reduction",
                    suggested=noise_reduction,
                    unit="strength",
                    description="Apply noise reduction"
                ))
            
            # Apply button
            if st.button("Apply Manual Adjustments"):
                # Apply the manual adjustments
                result = apply_adjustments(st.session_state.original_image, manual_adjustments)
                st.session_state.edited_image = result
                st.rerun()
        
        with tabs[2]:  # Style Presets tab
            st.subheader("Style Presets")
            
            # Get available styles
            available_styles = get_available_styles()
            
            # Create a radio button for each style
            style = st.radio(
                "Select a style preset:",
                available_styles,
                index=0
            )
            
            # Display style description
            st.info(get_style_description(style))
            
            # Apply button
            if st.button("Apply Style"):
                # Apply the selected style
                result = apply_adjustments(st.session_state.original_image, [], style=style)
                st.session_state.edited_image = result
                st.rerun()
        
        with tabs[3]:  # AI Assistant tab
            st.subheader("AI Photo Assistant")
            
            # Create subtabs for AI features
            ai_tabs = st.tabs(["Image Understanding", "Natural Language Editing", "Style Recommendations"])
            
            with ai_tabs[0]:  # Image Understanding
                st.write("Analyze your image using AI to understand its content and get smart adjustment recommendations.")
                
                if st.button("Analyze Image Content", key="analyze_btn"):
                    with st.spinner("Analyzing image content..."):
                        # Run AI analysis
                        adjustments, analysis = st.session_state.ai_analyzer.analyze(st.session_state.original_image)
                        st.session_state.ai_analysis = analysis
                        st.session_state.ai_adjustments = adjustments
                        
                        # Force a rerun to show results
                        st.rerun()
                
                # Display analysis results if available
                if 'ai_analysis' in st.session_state:
                    st.subheader("Image Analysis Results")
                    
                    # Display scene type
                    scene_type = st.session_state.ai_analysis.get('scene_type', 'unknown')
                    st.write(f"**Scene Type:** {scene_type.capitalize()}")
                    
                    # Display lighting condition
                    lighting = st.session_state.ai_analysis.get('lighting_condition', 'normal')
                    st.write(f"**Lighting Condition:** {lighting.replace('_', ' ').capitalize()}")
                    
                    # Display face detection results
                    has_faces = st.session_state.ai_analysis.get('has_faces', False)
                    face_count = st.session_state.ai_analysis.get('face_count', 0)
                    if has_faces:
                        st.write(f"**Detected {face_count} {'face' if face_count == 1 else 'faces'}**")
                    
                    # Display detected objects
                    objects = st.session_state.ai_analysis.get('objects', [])
                    if objects:
                        st.write("**Detected Objects:**")
                        for obj in objects:
                            conf = obj.get('confidence', 0) * 100
                            st.write(f"- {obj['class'].capitalize()} ({conf:.0f}%)")
                    
                    # Display color palette
                    palette = st.session_state.ai_analysis.get('color_palette', [])
                    if palette:
                        st.write("**Color Palette:**")
                        palette_cols = st.columns(len(palette))
                        for i, color in enumerate(palette):
                            with palette_cols[i]:
                                # Display color swatch
                                st.markdown(
                                    f"""
                                    <div style="background-color: rgb{tuple(color)}; 
                                    width: 50px; height: 50px; border-radius: 5px;">
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                    
                    # Display AI-recommended adjustments
                    if 'ai_adjustments' in st.session_state and st.session_state.ai_adjustments:
                        st.subheader("AI-Recommended Adjustments")
                        
                        # Create a checkbox for each adjustment
                        selected_ai_adjustments = []
                        for i, adj in enumerate(st.session_state.ai_adjustments):
                            if st.checkbox(
                                f"{adj.description} ({adj.suggested} {adj.unit})", 
                                value=True,
                                key=f"ai_adj_{i}"
                            ):
                                selected_ai_adjustments.append(adj)
                        
                        # Apply button
                        if st.button("Apply AI Adjustments"):
                            # Apply the selected adjustments
                            result = apply_adjustments(st.session_state.original_image, selected_ai_adjustments)
                            st.session_state.edited_image = result
                            st.rerun()
            
            with ai_tabs[1]:  # Natural Language Editing
                st.write("""Describe how you want to edit the image in plain English.
                The AI will understand your description and apply the appropriate adjustments.""")
                
                # Add examples for user reference
                with st.expander("Show Examples"):
                    st.write("- 'Make the image brighter and increase contrast'")
                    st.write("- 'Add a cinematic look with warm colors'")
                    st.write("- 'Create a dramatic black and white film noir style'")
                    st.write("- 'Enhance the colors and make them more vibrant'")
                    st.write("- 'Reduce noise and sharpen the details'")
                
                # Text input for natural language instructions
                nl_instruction = st.text_area(
                    "Enter your editing instructions:",
                    placeholder="e.g., Make the image warmer and increase contrast slightly",
                    key="nl_input"
                )
                
                # Process button
                if st.button("Process Natural Language"):
                    if not nl_instruction:
                        st.warning("Please enter some instructions.")
                    else:
                        with st.spinner("Processing natural language instructions..."):
                            # Process the natural language instruction
                            result, metadata = st.session_state.nl_processor.process(
                                st.session_state.original_image, 
                                nl_instruction
                            )
                            
                            # Update the edited image
                            st.session_state.edited_image = result
                            
                            # Store metadata for display
                            st.session_state.nl_metadata = metadata
                            
                            # Refresh the display
                            st.rerun()
                
                # Display function call info if available
                if 'nl_metadata' in st.session_state:
                    # Show what functions were called
                    st.subheader("Applied Operations")
                    
                    for func_call in st.session_state.nl_metadata['functions_called']:
                        func_name = func_call['name']
                        args = func_call['args']
                        
                        if func_name == "adjust_exposure":
                            direction = "increased" if args['amount'] > 0 else "decreased"
                            st.write(f"- {direction.capitalize()} brightness by {abs(args['amount']):.2f} EV")
                        elif func_name == "adjust_contrast":
                            direction = "increased" if args['multiplier'] > 1 else "decreased"
                            amount = abs(args['multiplier'] - 1) * 100
                            st.write(f"- {direction.capitalize()} contrast by {amount:.0f}%")
                        elif func_name == "adjust_saturation":
                            direction = "increased" if args['adjustment'] > 0 else "decreased"
                            st.write(f"- {direction.capitalize()} saturation by {abs(args['adjustment']):.2f}")
                        elif func_name == "adjust_temperature":
                            direction = "warmer" if args['adjustment'] > 0 else "cooler"
                            st.write(f"- Made colors {direction} by {abs(args['adjustment']):.2f}")
                        elif func_name == "adjust_sharpness":
                            st.write(f"- Applied sharpening with strength {args['strength']:.2f}")
                        elif func_name == "reduce_noise":
                            st.write(f"- Applied noise reduction with strength {args['strength']:.2f}")
                        elif func_name == "apply_style":
                            st.write(f"- Applied '{args['style_name']}' style")
                    
                    # Show any errors
                    if st.session_state.nl_metadata['errors']:
                        st.error("Errors encountered:")
                        for error in st.session_state.nl_metadata['errors']:
                            st.write(f"- {error}")
            
            with ai_tabs[2]:  # Style Recommendations
                st.write("""The AI will analyze your image and recommend suitable style presets,
                or you can describe the style you want to achieve.""")
                
                # Two options: content-based or description-based
                recommendation_mode = st.radio(
                    "Recommendation mode:",
                    ["Based on image content", "Based on your description"],
                    key="recommendation_mode"
                )
                
                if recommendation_mode == "Based on image content":
                    # Content-based recommendation
                    if st.button("Get Style Recommendations"):
                        with st.spinner("Analyzing image and finding styles..."):
                            # Get style recommendations
                            recommendations = st.session_state.rag_style_engine.recommend_style(
                                st.session_state.original_image
                            )
                            
                            # Store recommendations
                            st.session_state.style_recommendations = recommendations
                            
                            # Refresh display
                            st.rerun()
                else:
                    # Description-based recommendation
                    style_description = st.text_area(
                        "Describe the style you want:",
                        placeholder="e.g., Like a dramatic movie scene with high contrast",
                        key="style_description"
                    )
                    
                    if st.button("Find Matching Styles"):
                        if not style_description:
                            st.warning("Please enter a style description.")
                        else:
                            with st.spinner("Finding styles matching your description..."):
                                # Get style recommendations based on description
                                recommendations = st.session_state.rag_style_engine.recommend_style(
                                    st.session_state.original_image,
                                    description=style_description
                                )
                                
                                # Store recommendations
                                st.session_state.style_recommendations = recommendations
                                
                                # Refresh display
                                st.rerun()
                
                # Display style recommendations if available
                if 'style_recommendations' in st.session_state and st.session_state.style_recommendations:
                    st.subheader("Recommended Styles")
                    
                    for i, rec in enumerate(st.session_state.style_recommendations):
                        with st.container():
                            st.markdown(f"### {i+1}. {rec['style']}")
                            st.write(rec['description'])
                            
                            # Show reasoning
                            with st.expander("Why this style?"):
                                for reason in rec['reasoning']:
                                    st.write(f"- {reason}")
                            
                            # Apply button for this style
                            if st.button(f"Apply {rec['style']}", key=f"apply_rec_{i}"):
                                with st.spinner(f"Applying {rec['style']}..."):
                                    # Apply the style
                                    result = st.session_state.rag_style_engine.apply_style(
                                        st.session_state.original_image,
                                        rec['style']
                                    )
                                    
                                    # Update the edited image
                                    st.session_state.edited_image = result
                                    
                                    # Refresh display
                                    st.rerun()
        
        # Reset button
        if st.sidebar.button("Reset Image"):
            # Reset the edited image to the original
            st.session_state.edited_image = st.session_state.original_image.copy()
            st.rerun()
        
        # Download button
        if st.sidebar.download_button(
            "Download Edited Image",
            data=Image.fromarray(st.session_state.edited_image).tobytes(),
            file_name="edited_" + uploaded_file.name,
            mime=f"image/{uploaded_file.name.split('.')[-1]}"
        ):
            st.sidebar.success("Image downloaded successfully!")
    else:
        # No image uploaded yet
        st.info("Please upload an image to begin.")

if __name__ == "__main__":
    main()
