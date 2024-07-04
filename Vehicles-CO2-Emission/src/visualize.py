import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

class EDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def desc_statistics(self):
        """Display descriptive statistics of the DataFrame."""
        st.write('**Descriptive statistics**')
        st.write(self.data.describe())
        
    def correlation_heatmap(self, cmap='coolwarm'):
        """Display correlation heatmap of the DataFrame."""
        st.write('**Correlation Heatmap**')
        fig, ax = plt.subplots()
        sns.heatmap(self.data.corr(), annot=True, cmap=cmap, ax=ax)
        st.pyplot(fig)
        
    def hist_plot(self, column: str, bins=30, kde=True):
        """Display histogram of a specified column."""
        st.write(f'**Histogram of {column}**')
        fig, ax = plt.subplots(figsize=(10,7))
        sns.histplot(self.data[column], bins=bins, kde=kde, ax=ax)
        st.pyplot(fig)
        
    def scatter_plot(self, x: str, y: str):
        """Display scatter plot for specified columns."""
        st.write(f'**Scatter Plot: {x} vs {y}**')
        fig, ax = plt.subplots(figsize=(12,8))
        sns.scatterplot(data=self.data, x=x, y=y, ax=ax)
        st.pyplot(fig)
        
    def box_plot(self, column: str):
        """Display box plot of a specified column."""
        st.write(f'**Box Plot of {column}**')
        fig, ax = plt.subplots(figsize=(12,8))
        sns.boxplot(y=self.data[column], ax=ax)
        st.pyplot(fig)
        
    def show_plots(self):
        """Show all plots for the DataFrame."""
        self.desc_statistics()
        self.correlation_heatmap('seismic')
    
        if 'Mean_CO2_(g/mi)' in self.data.columns:  
            self.hist_plot('Mean_CO2_(g/mi)')
            self.box_plot('Mean_CO2_(g/mi)')
        if 'Horsepower_(HP)' in self.data.columns and 'Weight_(lbs)' in self.data.columns: 
            self.scatter_plot('Horsepower_(HP)', 'Weight_(lbs)')
        

