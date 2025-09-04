from IPython.display import display, Image
from PIL import Image
import io

def save_graph(supervisor, filename="lang_graph.png"):
    img_bytes = supervisor.get_graph().draw_mermaid_png()
    lang_graph_img = Image.open(io.BytesIO(img_bytes))
    lang_graph_img.save("lang_graph.png")