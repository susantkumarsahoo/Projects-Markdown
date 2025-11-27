# Comprehensive Streamlit Techniques Reference

## Core Concepts

### Script Execution Model
- **Script Rerunning**: Entire Python script re-executes top-to-bottom on user interaction or code changes
- **Data Flow**: Sequential execution model - order matters

### Performance Optimization
- **`st.cache_data`**: Cache data objects (DataFrames, lists, dicts) to avoid recomputation
- **`st.cache_resource`**: Cache global resources (ML models, database connections)
- **Widget Keys**: Use `key` parameter for unique widget identity and state control

### State Management
- **`st.session_state`**: Dictionary-like object preserving variables across reruns
- **Callbacks**: Use `on_change`, `args`, `kwargs` for widget event handling

---

## Page Configuration

### Setup
- **`st.set_page_config`**: Configure page title, icon, layout, sidebar state
- **`st.title`**: Main page title
- **`st.header`**: Section headers
- **`st.subheader`**: Subsection headers

### Text Display
- **`st.write`**: Universal display function (auto-detects content type)
- **`st.markdown`**: Markdown-formatted text with HTML support
- **`st.caption`**: Small descriptive text
- **`st.code`**: Display code blocks with syntax highlighting
- **`st.latex`**: Render LaTeX mathematical equations
- **`st.text`**: Fixed-width plain text
- **`st.divider`**: Visual horizontal separator

---

## Layout System

### Containers
- **`st.container`**: Grouping element, can be modified later
- **`st.empty`**: Placeholder for dynamic content updates
- **`st.sidebar`**: Collapsible sidebar for controls/navigation

### Multi-Column Layouts
- **`st.columns`**: Create vertical column divisions
  ```python
  col1, col2, col3 = st.columns(3)
  with col1:
      st.write("Column 1")
  ```

### Organizational Components
- **`st.tabs`**: Tabbed interface for content organization
- **`st.expander`**: Collapsible sections with expand/collapse control
- **`st.popover`**: Popup overlay content
- **`st.dialog`**: Modal dialog boxes
- **`st.fragment`**: Isolated component with separate rerun behavior

### Navigation
- **`st.navigation`**: New navigation API for multi-page apps
- **Multi-page apps**: Use `pages/` folder structure for automatic routing

---

## Input Widgets

### Buttons & Toggles
- **`st.button`**: Clickable button (returns True on click)
- **`st.link_button`**: Button linking to external URL
- **`st.download_button`**: Download file trigger
- **`st.checkbox`**: Boolean selection checkbox
- **`st.toggle`**: Toggle switch for on/off states

### Selection Widgets
- **`st.radio`**: Single selection from radio button group
- **`st.selectbox`**: Dropdown single selection
- **`st.multiselect`**: Multiple item selection from list

### Slider Inputs
- **`st.slider`**: Numeric range slider
- **`st.select_slider`**: Ordered option slider

### Text Inputs
- **`st.text_input`**: Single-line text entry
- **`st.text_area`**: Multi-line text entry
- **`st.number_input`**: Numeric input with increment/decrement

### Date & Time
- **`st.date_input`**: Date picker calendar
- **`st.time_input`**: Time selection widget

### File & Media
- **`st.file_uploader`**: Upload files (supports multiple formats)
- **`st.camera_input`**: Capture image from user's camera
- **`st.color_picker`**: Color selection tool

---

## Forms

### Batch Input Handling
- **`st.form`**: Group widgets to prevent individual reruns
- **`st.form_submit_button`**: Submit all form values at once
  ```python
  with st.form("my_form"):
      name = st.text_input("Name")
      age = st.number_input("Age")
      submitted = st.form_submit_button("Submit")
  ```

---

## Data Display

### Tables & DataFrames
- **`st.dataframe`**: Interactive scrollable DataFrame display
- **`st.table`**: Static non-interactive table
- **`st.data_editor`**: Editable data grid (formerly experimental)

### Metrics & KPIs
- **`st.metric`**: Display metrics with optional delta/change indicator
  ```python
  st.metric("Revenue", "$1.2M", "+15%")
  ```

### Structured Data
- **`st.json`**: Pretty-print JSON with collapsible structure

---

## Charts & Visualization

### Built-in Charts
- **`st.line_chart`**: Line chart from DataFrame
- **`st.area_chart`**: Area/stacked chart
- **`st.bar_chart`**: Vertical bar chart
- **`st.map`**: Geographic scatter plot on map

### External Library Support
- **`st.pyplot`**: Matplotlib figures
- **`st.altair_chart`**: Altair/Vega-Lite charts
- **`st.plotly_chart`**: Plotly interactive charts
- **`st.bokeh_chart`**: Bokeh visualizations
- **`st.pydeck_chart`**: PyDeck 3D maps
- **`st.vega_lite_chart`**: Direct Vega-Lite specs

### Supported Libraries
- Matplotlib, Seaborn, Plotly, Altair, Bokeh, PyDeck, ECharts, Vega-Lite

---

## Media Elements

### Display Media
- **`st.image`**: Display images (PIL, numpy arrays, URLs)
- **`st.audio`**: Audio player for audio files
- **`st.video`**: Video player for video files

---

## Status & Progress

### Loading Indicators
- **`st.spinner`**: Temporary loading message with spinner
  ```python
  with st.spinner("Loading..."):
      # long operation
  ```
- **`st.progress`**: Progress bar (0-100%)
- **`st.status`**: Multi-step progress tracker

### User Notifications
- **`st.toast`**: Temporary popup notification
- **`st.success`**: Green success message box
- **`st.error`**: Red error message box
- **`st.warning`**: Yellow warning message box
- **`st.info`**: Blue informational message box
- **`st.exception`**: Display exception with traceback

### Celebrations
- **`st.balloons`**: Balloon animation
- **`st.snow`**: Snowfall animation

---

## Chat & Conversational UI

### Chat Components
- **`st.chat_input`**: Chat-style text input at bottom
- **`st.chat_message`**: Styled message bubble (user/assistant)
  ```python
  with st.chat_message("user"):
      st.write("Hello!")
  ```

---

## Authentication (Beta)

### User Management
- **`st.login`**: Login interface
- **`st.logout`**: Logout functionality
- **`st.user`**: Access current user information

---

## Advanced Features

### Experimental/Legacy
- **`st.experimental_rerun`**: Force script rerun
- **`st.experimental_get_query_params`**: Get URL parameters
- **`st.experimental_set_query_params`**: Set URL parameters
- **`st.experimental_connection`**: Database connection helper

### Configuration
- **`.streamlit/config.toml`**: App-wide configuration file
  - Theme colors (primary, background, text)
  - Dark/light mode defaults
  - Layout width settings
  - Server port and configuration

---

## Deployment Strategies

### Hosting Options
- **Streamlit Community Cloud**: Free hosting for public apps
- **Docker**: Containerized deployment
- **Local production**: Run with production server
- **Nginx + Gunicorn**: Enterprise deployment setup

---

## Best Practices

### Performance Tips
1. Cache expensive computations with `@st.cache_data`
2. Cache resources (models, connections) with `@st.cache_resource`
3. Use `st.form` to batch inputs and reduce reruns
4. Leverage `st.session_state` for preserving data
5. Use `st.fragment` for isolated component updates

### Code Organization
1. Use multi-page structure for complex apps (`pages/` folder)
2. Give widgets unique `key` values for reliable state management
3. Place sidebar elements in `st.sidebar` context
4. Structure layouts with `st.columns` and `st.tabs`
5. Use `st.empty()` for dynamic content placeholders

### User Experience
1. Show loading states with `st.spinner` or `st.progress`
2. Provide feedback with `st.success`, `st.error`, `st.warning`
3. Use `st.expander` to hide advanced options
4. Display metrics with `st.metric` for quick insights
5. Implement forms for multi-field submissions

---

## Quick Reference: Common Patterns

### Pattern: File Upload & Processing
```python
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
```

### Pattern: Caching Data
```python
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()
```

### Pattern: Session State Counter
```python
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Increment"):
    st.session_state.count += 1

st.write(f"Count: {st.session_state.count}")
```

### Pattern: Multi-Column Layout
```python
col1, col2, col3 = st.columns(3)
col1.metric("Metric 1", "100", "+10%")
col2.metric("Metric 2", "200", "-5%")
col3.metric("Metric 3", "300", "+2%")
```

### Pattern: Form with Validation
```python
with st.form("user_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    submit = st.form_submit_button("Submit")
    
    if submit:
        if name and email:
            st.success("Form submitted!")
        else:
            st.error("Please fill all fields")
```

---

## Summary

This reference covers all essential Streamlit techniques organized by category. Use this for:
- Quick lookups during development
- Exam preparation and review
- Understanding Streamlit's architecture and patterns
- Building production-ready applications

**Key Takeaway**: Streamlit's power lies in its simplicity—script reruns, caching, and session state form the foundation for all interactive applications.