import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class EDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def desc_statistics(self):
        """Display descriptive statistics of the DataFrame."""
        st.write('**Descriptive statistics**')
        st.write(self.data.describe())

    def correlation_heatmap(self):
        """Display correlation heatmap of the DataFrame using Plotly Express."""
        numeric_data = self.data.select_dtypes(include=['number'])
        correlation_matrix = numeric_data.corr()
        fig = px.imshow(correlation_matrix, 
                        labels=dict(color="Correlation"),
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        title="Correlation Heatmap",
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        width=1320, height=840)
        st.plotly_chart(fig)

    def hist_plot(self, column: str, bins=30):
        """Display histogram of a specified column."""
        
        fig = px.histogram(self.data, x=column, nbins=bins, title=f'Histogram of {column}')
        st.plotly_chart(fig)

    def scatter_plot(self, x: str, y: str):
        """Display scatter plot for specified columns."""
        
        fig = px.scatter(self.data, x=x, y=y, title=f'Scatter Plot: {x} vs {y}')
        st.plotly_chart(fig)

    def box_plot(self, column: str):
        """Display box plot of a specified column."""
        
        fig = px.box(self.data, y=column, title=f'Box Plot of {column}')
        st.plotly_chart(fig)

    def bar_plot_average_hp_by_manufacturer(self):
        """Bar plot of Average Horsepower by Manufacturer."""
        
        average_hp_by_manufacturer = self.data.groupby('Manufacturer')['Horsepower_(HP)'].mean().reset_index()
        fig = px.bar(average_hp_by_manufacturer, x='Manufacturer', y='Horsepower_(HP)', title='Average Horsepower by Manufacturer')
        st.plotly_chart(fig)

    def pie_chart_vehicle_type_distribution(self):
        """Pie chart of Vehicle Type Distribution."""
        
        vehicle_type_distribution = self.data['Vehicle_Type'].value_counts().reset_index()
        vehicle_type_distribution.columns = ['Vehicle_Type', 'Count']
        fig = px.pie(vehicle_type_distribution, values='Count', names='Vehicle_Type', title='Vehicle Type Distribution', hole=0.3)
        st.plotly_chart(fig)

    def stacked_bar_drivetrain_by_manufacturer(self):
        """Stacked bar chart of Drivetrain Distribution by Manufacturer."""
        
        drivetrain_columns = ['Drivetrain_Front', 'Drivetrain_4WD', 'Drivetrain_Rear']
        drivetrain_distribution = self.data.groupby('Manufacturer')[drivetrain_columns].sum().reset_index()
        fig = go.Figure()
        for col in drivetrain_columns:
            fig.add_trace(go.Bar(
                x=drivetrain_distribution['Manufacturer'],
                y=drivetrain_distribution[col],
                name=col
            ))
        fig.update_layout(barmode='stack', title='Drivetrain Distribution by Manufacturer')
        st.plotly_chart(fig)

    def violin_plot_mpg_by_regulatory_class(self):
        """Violin plot of MPG by Regulatory Class."""
        
        fig = px.violin(self.data, x="Regulatory_Class", y="Mean_Real_World_MPG", box=True, title='Violin Plot of MPG by Regulatory Class')
        st.plotly_chart(fig)

    def bar_plot_co2_by_transmission_type(self):
        """Bar plot of Average CO2 Emissions by Transmission Type."""
        
        transmission_columns = ['Transmission_Manual', 'Transmission_Automatic', 'Transmission_Lockup', 'Transmission_CVT_(Hybrid)', 'Transmission_Other']
        data_melted = self.data.melt(id_vars=['Mean_CO2_(g/mi)'], value_vars=transmission_columns, var_name='Transmission_Type', value_name='Presence')
        data_melted = data_melted[data_melted['Presence'] == 1]
        average_co2_by_transmission = data_melted.groupby('Transmission_Type')['Mean_CO2_(g/mi)'].mean().reset_index()
        fig = px.bar(average_co2_by_transmission, x='Transmission_Type', y='Mean_CO2_(g/mi)', title='Average CO2 Emissions by Transmission Type')
        st.plotly_chart(fig)

    def show_plots(self):
        """Show all plots for the DataFrame."""
        self.desc_statistics()
        self.correlation_heatmap()

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
