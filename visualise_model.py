import torch
from torchviz import make_dot
from torchsummary import summary


def visualize_model(model, input_size, output_path="model_structure.png"):
    """
    Visualize and print the structure of any PyTorch model.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to visualize.
    - input_size (tuple): The shape of the input tensor (e.g., (3, 224, 224) for an RGB image).
    - output_path (str): Path to save the block diagram image (default: "model_structure.png").
    """
    # Generate a random input tensor with the specified size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = torch.randn(1, *input_size).to(device)

    print("Model Summary:")
    summary(model, input_size=input_size)

    y = model(dummy_input)
    graph = make_dot(y, params=dict(model.named_parameters()))
    graph.render(output_path, format="png")
    print(f"Model diagram saved to: {output_path}")


if __name__ == "__main__":
    from srresnet import _NetG

    model = _NetG()
    visualize_model(model, input_size=(3, 224, 224), output_path="netg_structure")
