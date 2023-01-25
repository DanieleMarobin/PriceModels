import streamlit as st
from bokeh.models.widgets import MultiSelect
from bokeh.layouts import column

# Create a list of options for the MultiSelect widget
options = ['Option 1', 'Option 2', 'Option 3']
options = [f'Option {i}' for i in range(10)]

# Create the MultiSelect widget
multiselect = MultiSelect(title="Select Options:", value=[], options=options, size = 10)

# Use st.bokeh_chart to render the widget in the Streamlit app
# st.bokeh_chart(column(multiselect))
st.bokeh_chart(multiselect)

# Get the selected options from the MultiSelect widget
selected_options = multiselect.value
st.write("Selected options:", selected_options)