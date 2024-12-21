
# *** modules/utils.py ***
import plotly.express as px


def plot_2d_scatter(x, color, title, symbol):
    # Create a scatter plot
    fig = px.scatter(
        x=x[:, 0],
        y=x[:, 1],
        color=color,  # Assign colors based on target values
        labels={'color': 'Class'},
        title=title,
        opacity=0.8,
        symbol=symbol,
        symbol_sequence=['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star'],
    )
    # Update the layout for better visualization
    fig.update_layout(
        template='plotly_white',
        xaxis_title='feature 1',
        yaxis_title='feature 2',
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(title="Class", x=1.05),
        # legend=dict(x=0, y=1, traceorder="normal"),
        legend=dict(
            x=1.05,  # Position to the right
            y=0.5,  # Centered vertically
            borderwidth=1,
            # itemsizing='constant',
            traceorder="normal"
        ),
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Ensure x and y scales are the same
        yaxis=dict(scaleanchor="x"),
        width=1000,  # Set the width of the figure
        height=600,  # Set the height of the figure
        margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins
    )

    # Add marker border and update marker size
    fig.update_traces(
        marker=dict(size=15, line=dict(width=0.5, color='black'))  # Border color
    )

    # Show the plot
    # fig.show()

    return fig

