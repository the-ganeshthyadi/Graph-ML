import matplotlib.pyplot as plt
import io
import base64

def generate_graph(params):
    fig, ax = plt.subplots()

    # Example graph parameters
    graph_type = params.get('type', 'line')
    color = params.get('color', 'blue')
    linewidth = params.get('linewidth', 1)
    background_color = params.get('background_color', 'white')
    x = params.get('x', [1, 2, 3])
    y = params.get('y', [1, 4, 9])

    ax.set_facecolor(background_color)
    
    if graph_type == 'line':
        ax.plot(x, y, color=color, linewidth=linewidth)
    elif graph_type == 'scatter':
        ax.scatter(x, y, color=color)
    elif graph_type == 'bar':
        ax.bar(x, y, color=color)
    
    # Convert plot to PNG and encode as base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64
    