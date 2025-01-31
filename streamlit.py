import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Title and intro
st.title("Full-Fledged Streamlit App Example")
st.subheader("Interactive UI with file upload, data analysis, and visualization.")

# Sidebar with user input options
st.sidebar.title("Options")
st.sidebar.subheader("Choose your action")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# If file is uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(df.head())  # Display the first 5 rows of the dataset
else:
    # Load the iris dataset as a fallback if no file is uploaded
    st.write("No file uploaded. Displaying Iris dataset as default.")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    st.write(df.head())  # Show the first 5 rows

# Select column for analysis
column = st.selectbox("Select a column to analyze", df.columns)

# Display statistics for the selected column
st.write(f"### Statistics for {column}:")
st.write(df[column].describe())

# Display histogram for the selected column
st.write(f"### Histogram of {column}:")
fig, ax = plt.subplots()
df[column].hist(bins=20, ax=ax)
st.pyplot(fig)

# Create a correlation heatmap
st.write("### Correlation Heatmap:")
corr = df.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Checkbox to show a line plot for sepal width vs sepal length (from the Iris dataset)
if st.checkbox("Show Sepal Width vs Sepal Length Line Plot"):
    fig, ax = plt.subplots()
    ax.plot(df['sepal length (cm)'], df['sepal width (cm)'])
    ax.set_xlabel("Sepal Length (cm)")
    ax.set_ylabel("Sepal Width (cm)")
    ax.set_title("Sepal Length vs Sepal Width")
    st.pyplot(fig)

# Interactive slider for selecting the number of bins in the histogram
num_bins = st.slider("Select number of bins for histogram", 5, 50, 20)
st.write(f"Displaying histogram with {num_bins} bins")
fig, ax = plt.subplots()
df[column].hist(bins=num_bins, ax=ax)
st.pyplot(fig)

# Display the current time
st.write("### Current Time:")
st.write(pd.to_datetime('now'))

# Button to refresh the app (you can add your custom logic for it)
if st.button("Refresh"):
    st.experimental_rerun()

# Footer
st.write("Developed by Your Name")
