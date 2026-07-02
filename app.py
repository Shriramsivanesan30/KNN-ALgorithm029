# ============================================
# 🌸 Iris Flower Classification using kNN
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page settings
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# ============================================
# STEP 1: Load Data & Train Model
# ============================================

@st.cache_data
def get_model(k):
    """Load data and train kNN model"""
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features (important for kNN!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train kNN model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, iris, accuracy

# ============================================
# STEP 2: Main App
# ============================================

def main():
    # Title
    st.title("🌸 Iris Flower Classifier")
    st.write("Predict the species using k-Nearest Neighbors algorithm")
    st.divider()
    
    # Sidebar - k value selection
    st.sidebar.header("⚙️ Settings")
    k = st.sidebar.slider("Select k value", 1, 15, 5)
    st.sidebar.write(f"📌 k = {k} (number of neighbors)")
    
    # Get model
    model, scaler, iris, accuracy = get_model(k)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Model Info", "📖 Learn kNN"])
    
    # ============================================
    # TAB 1: Prediction
    # ============================================
    with tab1:
        st.header("Enter Flower Measurements")
        st.write("Adjust the sliders (measurements in cm):")
        
        # Input sliders
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1, 0.1)
            sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5, 0.1)
        
        with col2:
            petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4, 0.1)
            petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2, 0.1)
        
        # Predict button
        if st.button("🔮 Predict Species", type="primary"):
            # Prepare input
            user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Scale input
            user_input_scaled = scaler.transform(user_input)
            
            # Predict
            prediction = model.predict(user_input_scaled)
            probabilities = model.predict_proba(user_input_scaled)
            
            # Get species name
            species = iris.target_names[prediction[0]]
            
            # Show result
            st.divider()
            
            # Color based on species
            if species == "setosa":
                color = "🔴"
            elif species == "versicolor":
                color = "🟣"
            else:
                color = "🟢"
            
            st.success(f"{color} **Predicted Species: {species.upper()}**")
            
            # Show probabilities
            st.subheader("Probability Distribution")
            
            prob_data = {
                "Species": iris.target_names,
                "Probability (%)": [round(p * 100, 2) for p in probabilities[0]]
            }
            prob_df = pd.DataFrame(prob_data)
            st.bar_chart(prob_df, x="Species", y="Probability (%)")
            
            # Show detailed probabilities
            for name, prob in zip(iris.target_names, probabilities[0]):
                st.write(f"**{name.capitalize()}**: {prob*100:.1f}%")
    
    # ============================================
    # TAB 2: Model Info
    # ============================================
    with tab2:
        st.header("Model Performance")
        
        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy*100:.1f}%")
        col2.metric("k Value", k)
        
        st.divider()
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        # Get predictions for confusion matrix
        iris_data = load_iris()
        X_all = iris_data.data
        y_all = iris_data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        # Display confusion matrix as table
        cm_df = pd.DataFrame(
            cm,
            index=iris.target_names,
            columns=iris.target_names
        )
        cm_df.index.name = "Actual"
        cm_df.columns.name = "Predicted"
        st.dataframe(cm_df, use_container_width=True)
        
        st.divider()
        
        # Sample data
        st.subheader("Sample Data")
        sample_df = pd.DataFrame(
            iris_data.data[:10],
            columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        )
        sample_df['Species'] = [iris.target_names[i] for i in iris_data.target[:10]]
        st.dataframe(sample_df, use_container_width=True)
    
    # ============================================
    # TAB 3: Learn kNN
    # ============================================
    with tab3:
        st.header("How kNN Works")
        
        st.write("""
        ### 📌 What is kNN?
        **k-Nearest Neighbors** is a simple algorithm that:
        - Remembers all training data
        - For new data, finds 'k' closest examples
        - Predicts based on majority vote
        """)
        
        st.write("""
        ### 🔄 Steps:
        1. **Store** all training flowers with their species
        2. **Receive** new flower measurements
        3. **Calculate** distance to all training flowers
        4. **Find** k closest flowers (neighbors)
        5. **Vote** - which species is most common?
        6. **Predict** that species!
        """)
        
        st.write("""
        ### 📏 Distance Formula (Euclidean):
        ```
        Distance = √[(x₁-x₂)² + (y₁-y₂)² + ...]
        ```
        """)
        
        st.write("""
        ### ⚖️ Choosing k:
        | k Value | Effect |
        |---------|--------|
        | Small (1-3) | Sensitive to noise |
        | Medium (5-7) | Good balance ✅ |
        | Large (10+) | May miss patterns |
        """)
        
        # Simple visualization
        st.subheader("Visual Example")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        np.random.seed(42)
        
        # Two classes
        x1, y1 = np.random.normal(2, 0.5, 10), np.random.normal(2, 0.5, 10)
        x2, y2 = np.random.normal(5, 0.5, 10), np.random.normal(5, 0.5, 10)
        
        # Plot points
        ax.scatter(x1, y1, c='red', s=80, label='Setosa', edgecolors='black')
        ax.scatter(x2, y2, c='blue', s=80, label='Versicolor', edgecolors='black')
        
        # New point
        new_x, new_y = 3.5, 3.5
        ax.scatter(new_x, new_y, c='green', s=200, marker='*', label='New Flower', edgecolors='black', zorder=5)
        
        # Draw circle
        circle = plt.Circle((new_x, new_y), 1.8, fill=False, color='green', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_xlabel('Petal Length (cm)')
        ax.set_ylabel('Petal Width (cm)')
        ax.set_title(f'kNN with k={k}: Finding Nearest Neighbors')
        ax.legend()
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
