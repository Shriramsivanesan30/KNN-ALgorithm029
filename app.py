# ============================================
# 🌸 Iris Flower Classification using kNN
# A Simple Machine Learning Project with Streamlit
# ============================================

# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# --- Custom CSS for better styling ---
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #2c3e50;
        }
        .success-box {
            background-color: #d4edda;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .info-box {
            background-color: #d1ecf1;
            padding: 15px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================
# STEP 1: Load and Prepare Data
# ============================================

@st.cache_resource  # Cache the model to avoid retraining every time
def load_data_and_train_model(k_value):
    """
    Load iris dataset, preprocess, and train kNN model.
    Returns: model, scaler, X_test, y_test
    """
    # Load the built-in Iris dataset
    iris = load_iris()
    
    # Features: sepal length, sepal width, petal length, petal width
    X = iris.data
    # Target: 0=Setosa, 1=Versicolor, 2=Virginica
    y = iris.target
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features (important for kNN!)
    # kNN uses distance, so features should be on same scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train kNN model
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test, iris


# ============================================
# STEP 2: Build the Streamlit App
# ============================================

def main():
    # --- Title ---
    st.markdown("<h1 class='main-header'>🌸 Iris Flower Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#7f8c8d;'>Predict the species of an Iris flower using k-Nearest Neighbors</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # --- Sidebar for Settings ---
    st.sidebar.header("⚙️ Settings")
    k_value = st.sidebar.slider(
        "Select k (number of neighbors)",
        min_value=1,
        max_value=15,
        value=5,
        help="k is the number of nearest neighbors to consider for prediction"
    )
    
    # Show what k means
    st.sidebar.info(f"📌 k = {k_value}\n\nThe model will look at the {k_value} most similar flowers to make a prediction.")
    
    # --- Load Data and Train Model ---
    model, scaler, X_test, y_test, iris = load_data_and_train_model(k_value)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # --- Create Tabs for Organization ---
    tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Model Info", "📖 About kNN"])
    
    # ============================================
    # TAB 1: Make Predictions
    # ============================================
    with tab1:
        st.header("Enter Flower Measurements")
        st.markdown("Adjust the sliders to match your flower's measurements (in cm):")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.slider(
                "Sepal Length (cm)",
                min_value=4.0,
                max_value=8.0,
                value=5.1,
                step=0.1
            )
            sepal_width = st.slider(
                "Sepal Width (cm)",
                min_value=2.0,
                max_value=4.5,
                value=3.5,
                step=0.1
            )
        
        with col2:
            petal_length = st.slider(
                "Petal Length (cm)",
                min_value=1.0,
                max_value=7.0,
                value=1.4,
                step=0.1
            )
            petal_width = st.slider(
                "Petal Width (cm)",
                min_value=0.1,
                max_value=2.5,
                value=0.2,
                step=0.1
            )
        
        # Predict button
        if st.button("🔮 Predict Species", type="primary", use_container_width=True):
            # Create input array
            user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Scale the input (same as training data)
            user_input_scaled = scaler.transform(user_input)
            
            # Make prediction
            prediction = model.predict(user_input_scaled)
            probabilities = model.predict_proba(user_input_scaled)
            
            # Get the predicted class name
            species_name = iris.target_names[prediction[0]]
            
            # Display result
            st.divider()
            st.markdown("<h3 style='text-align:center;'>Prediction Result</h3>", unsafe_allow_html=True)
            
            # Show predicted species
            if species_name == "setosa":
                emoji = "🔴"
                color = "#e74c3c"
            elif species_name == "versicolor":
                emoji = "🟣"
                color = "#9b59b6"
            else:
                emoji = "🟢"
                color = "#27ae60"
            
            st.markdown(f"""
                <div class='success-box'>
                    <h2 style='color:{color};'>{emoji} {species_name.upper()}</h2>
                    <p>The flower is predicted to be <strong>Iris {species_name}</strong></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show probability distribution
            st.subheader("Probability Distribution")
            
            # Create a dataframe for probabilities
            prob_df = pd.DataFrame({
                'Species': iris.target_names,
                'Probability': probabilities[0] * 100
            })
            
            # Display as bar chart
            fig, ax = plt.subplots(figsize=(8, 3))
            colors = ['#e74c3c', '#9b59b6', '#27ae60']
            bars = ax.barh(prob_df['Species'], prob_df['Probability'], color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability (%)')
            ax.set_title('Prediction Confidence')
            
            # Add percentage labels on bars
            for bar, prob in zip(bars, prob_df['Probability']):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{prob:.1f}%', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # ============================================
    # TAB 2: Model Information
    # ============================================
    with tab2:
        st.header("Model Performance")
        
        # Show accuracy
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy*100:.1f}%")
        with col2:
            st.metric("k Value", k_value)
        with col3:
            st.metric("Test Samples", len(y_test))
        
        st.divider()
        
        # Show confusion matrix
        st.subheader("Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.divider()
        
        # Show sample data
        st.subheader("Sample Data from Dataset")
        
        # Create a dataframe to display
        sample_df = pd.DataFrame(
            iris.data,
            columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        )
        sample_df['Species'] = [iris.target_names[i] for i in iris.target]
        
        st.dataframe(sample_df.head(10), use_container_width=True)
        
        # Feature visualization
        st.subheader("Feature Distribution by Species")
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        colors = {'setosa': '#e74c3c', 'versicolor': '#9b59b6', 'virginica': '#27ae60'}
        
        for idx, (feature, ax) in enumerate(zip(features, axes.flatten())):
            for species in iris.target_names:
                data = sample_df[sample_df['Species'] == species][feature]
                ax.hist(data, alpha=0.6, label=species, color=colors[species], bins=15)
            ax.set_xlabel(feature + ' (cm)')
            ax.set_ylabel('Count')
            ax.legend()
        
        plt.suptitle('Distribution of Features by Species', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
    
    # ============================================
    # TAB 3: About kNN
    # ============================================
    with tab3:
        st.header("How does kNN work?")
        
        st.markdown("""
        ### 📌 What is kNN?
        **k-Nearest Neighbors (kNN)** is a simple machine learning algorithm that:
        - Stores all training data
        - For new data, finds the 'k' closest training examples
        - Predicts based on majority vote of those neighbors
        
        ---
        
        ### 🔄 Step-by-Step Process
        
        ```
        1. Load training data (flowers with known species)
        
        2. Receive new flower measurements
        
        3. Calculate distance to ALL training flowers
        
        4. Find the k closest flowers (neighbors)
        
        5. Count which species appears most among neighbors
        
        6. Predict that species!
        ```
        
        ---
        
        ### 📏 Distance Calculation
        kNN typically uses **Euclidean distance**:
        
        ```
        Distance = √((x₁-x₂)² + (y₁-y₂)² + ...)
        ```
        
        ---
        
        ### ⚖️ Choosing k
        | k Value | Effect |
        |---------|--------|
        | Small (1-3) | More sensitive to noise |
        | Medium (5-7) | Good balance (recommended) |
        | Large (10+) | Smoother but may miss patterns |
        
        ---
        
        ### ✅ Advantages
        - Simple to understand
        - No training phase needed
        - Works well with small datasets
        
        ### ❌ Disadvantages
        - Slow prediction (calculates all distances)
        - Sensitive to feature scaling
        - Doesn't work well with high dimensions
        """)
        
        # Simple visual explanation
        st.subheader("Visual Example")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create simple 2D example
        np.random.seed(42)
        
        # Class 1 (red dots)
        x1 = np.random.normal(2, 0.5, 10)
        y1 = np.random.normal(2, 0.5, 10)
        
        # Class 2 (blue dots)
        x2 = np.random.normal(5, 0.5, 10)
        y2 = np.random.normal(5, 0.5, 10)
        
        # New point
        new_x, new_y = 3.5, 3.5
        
        # Plot training points
        ax.scatter(x1, y1, c='red', s=100, label='Setosa', edgecolors='black', zorder=3)
        ax.scatter(x2, y2, c='blue', s=100, label='Versicolor', edgecolors='black', zorder=3)
        
        # Plot new point
        ax.scatter(new_x, new_y, c='green', s=200, marker='*', label='New Flower', edgecolors='black', zorder=4)
        
        # Draw lines to nearest neighbors (for illustration)
        all_x = np.concatenate([x1, x2])
        all_y = np.concatenate([y1, y2])
        distances = np.sqrt((all_x - new_x)**2 + (all_y - new_y)**2)
        nearest_indices = np.argsort(distances)[:k_value]
        
        for idx in nearest_indices:
            ax.plot([new_x, all_x[idx]], [new_y, all_y[idx]], 
                   'g--', alpha=0.5, linewidth=1.5)
        
        # Draw circle around k neighbors
        circle = plt.Circle((new_x, new_y), distances[nearest_indices[-1]], 
                           fill=False, color='green', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_xlabel('Petal Length (cm)', fontsize=12)
        ax.set_ylabel('Petal Width (cm)', fontsize=12)
        ax.set_title(f'kNN with k={k_value}: Finding {k_value} Nearest Neighbors', fontsize=14)
        ax.legend(loc='upper left')
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)


# ============================================
# Run the App
# ============================================

if __name__ == "__main__":
    main()
