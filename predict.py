import torch
from model import SimpleRegressionModel

def load_model(model_path="liner_regression_model.pth"):
    # loads the trained model & scalling parameters
    checkpoint = torch.load(model_path)

    model = SimpleRegressionModel()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # VERY IMPORTANT: inference mode

    X_mean = checkpoint["X_mean"]
    X_std = checkpoint["X_std"]
    y_mean = checkpoint["y_mean"]
    y_std = checkpoint["y_std"]

    return model, X_mean, X_std, y_mean, y_std



def predict(value, model, X_mean, X_std, y_mean, y_std):
    """
    Takes a raw input value and returns predicted output
    """
    # Convert to tensor
    input_tensor = torch.tensor([[value]], dtype=torch.float32)

    # Scale input
    input_scaled = (input_tensor - X_mean) / X_std

    # Predict (scaled)
    with torch.no_grad():  # No training, no gradients
        prediction_scaled = model(input_scaled)

    # Unscale output
    prediction_real = prediction_scaled * y_std + y_mean

    return prediction_real.item()

if __name__ == "__main__":
    # Load trained model
    model, X_mean, X_std, y_mean, y_std = load_model()

    print("Model loaded successfully!")

    # Test predictions
    test_inputs = [1.0, 5.0, 10.0]

    for val in test_inputs:
        output = predict(val, model, X_mean, X_std, y_mean, y_std)
        print(f"Input: {val} → Predicted Output: {output:.2f}")