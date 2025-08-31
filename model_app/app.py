# Main app file for Hugging Face Spaces deployment
from dotenv import load_dotenv

load_dotenv()

from app.gradio_interface import create_interface

# Create and launch the interface
app = create_interface()

# Launch for Hugging Face Spaces
if __name__ == "__main__":
    # queue() improves responsiveness for heavier explainers
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)