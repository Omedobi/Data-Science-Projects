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
        fig, ax = plt.subplots(figsize=(30,25))
        numeric_data = self.data.select_dtypes(include=['number'])
        sns.heatmap(numeric_data.corr(), linewidths=0.2, annot=True, fmt=".2f", cmap=cmap, ax=ax)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        st.pyplot(fig)
        
    def hist_plot(self, column: str, bins=30, kde=True):
        """Display histogram of a specified column."""
        st.write(f'**Histogram of {column}**')
        fig, ax = plt.subplots(figsize=(12,8))
        sns.histplot(self.data[column], bins=bins, kde=kde, ax=ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        st.pyplot(fig)
        
    def scatter_plot(self, x: str, y: str):
        """Display scatter plot for specified columns."""
        st.write(f'**Scatter Plot: {x} vs {y}**')
        fig, ax = plt.subplots(figsize=(12,8))
        sns.scatterplot(data=self.data, x=x, y=y, ax=ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        st.pyplot(fig)
        
    def box_plot(self, column: str):
        """Display box plot of a specified column."""
        st.write(f'**Box Plot of {column}**')
        fig, ax = plt.subplots(figsize=(12,8))
        sns.boxplot(y=self.data[column], ax=ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        st.pyplot(fig)

    def bar_plot_average_hp_by_manufacturer(self):
        """Bar plot of Average Horsepower by Manufacturer."""
        st.write('**Average Horsepower by Manufacturer**')
        average_hp_by_manufacturer = self.data.groupby('Manufacturer')['Horsepower_(HP)'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12,8))
        sns.barplot(x='Horsepower_(HP)', y='Manufacturer', data=average_hp_by_manufacturer, ax=ax)
        plt.title('Average Horsepower by Manufacturer')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        st.pyplot(fig)

    def pie_chart_vehicle_type_distribution(self):
        """Pie chart of Vehicle Type Distribution."""
        st.write('**Vehicle Type Distribution**')
        vehicle_type_distribution = self.data['Vehicle_Type'].value_counts()
        fig, ax = plt.subplots(figsize=(12,8))
        ax.pie(vehicle_type_distribution, labels=vehicle_type_distribution.index, autopct='%1.1f%%', startangle=140)
        plt.title('Vehicle Type Distribution')
        st.pyplot(fig)

    def stacked_bar_drivetrain_by_manufacturer(self):
        """Stacked bar chart of Drivetrain Distribution by Manufacturer."""
        st.write('**Drivetrain Distribution by Manufacturer**')
        drivetrain_columns = ['Drivetrain_Front', 'Drivetrain_4WD', 'Drivetrain_Rear']
        drivetrain_distribution = self.data.groupby('Manufacturer')[drivetrain_columns].sum()
        fig, ax = plt.subplots(figsize=(12,8))
        drivetrain_distribution.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        plt.title('Drivetrain Distribution by Manufacturer')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        st.pyplot(fig)

    def violin_plot_mpg_by_regulatory_class(self):
        """Violin plot of MPG by Regulatory Class."""
        st.write('**Violin Plot of MPG by Regulatory Class**')
        fig, ax = plt.subplots(figsize=(12,8))
        sns.violinplot(x="Regulatory_Class", y="Mean_Real_World_MPG", data=self.data, ax=ax)
        plt.title('Violin Plot of MPG by Regulatory Class')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        st.pyplot(fig)

    def bar_plot_co2_by_transmission_type(self):
        """Bar plot of Average CO2 Emissions by Transmission Type."""
        st.write('**Average CO2 Emissions by Transmission Type**')
        transmission_columns = ['Transmission_Manual', 'Transmission_Automatic', 'Transmission_Lockup', 'Transmission_CVT_(Hybrid)', 'Transmission_Other']
        data_melted = self.data.melt(id_vars=['Mean_CO2_(g/mi)'], value_vars=transmission_columns, var_name='Transmission_Type', value_name='Presence')
        data_melted = data_melted[data_melted['Presence'] == 1]
        average_co2_by_transmission = data_melted.groupby('Transmission_Type')['Mean_CO2_(g/mi)'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12,8))
        sns.barplot(x='Mean_CO2_(g/mi)', y='Transmission_Type', data=average_co2_by_transmission, ax=ax)
        plt.title('Average CO2 Emissions by Transmission Type')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
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
        
        self.bar_plot_average_hp_by_manufacturer()
        self.pie_chart_vehicle_type_distribution()
        self.stacked_bar_drivetrain_by_manufacturer()
        self.violin_plot_mpg_by_regulatory_class()
        self.bar_plot_co2_by_transmission_type()
