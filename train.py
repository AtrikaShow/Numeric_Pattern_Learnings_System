import torch

from model import SimpleRegressionModel
from data_generator import generate_data 

def main():
    # generate data
    input_data, output_data = generate_data(50)

        # Convert Python lists to PyTorch tensors
    # Shape: (batch_size, features)
    X_tensor = torch.tensor(input_data, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(output_data, dtype=torch.float32).view(-1, 1)

    print("Sample input data:", X_tensor[:5])
    print("Sample output data:", y_tensor[:5])

    # -------------------------------
    # 2. Initialize Model
    # -------------------------------
    model = SimpleRegressionModel()

    # -------------------------------
    # 3. Define Loss Function & Optimizer
    # -------------------------------
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # -------------------------------
    # 4. Training Loop
    # -------------------------------
    epochs = 500

    for epoch in range(epochs):
        # Forward pass
        predictions = model(X_tensor)

        # Compute loss
        loss = loss_fn(predictions, y_tensor)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Logging
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # -------------------------------
    # 5. Test the Trained Model
    # -------------------------------
    test_values = [1.0, 5.0, 10.0]

    print("\nTesting trained model:")
    for value in test_values:
        test_tensor = torch.tensor([[value]])
        prediction = model(test_tensor)
        print(f"Input: {value} → Predicted Output: {prediction.item():.2f}")


if __name__ == "__main__":
    main()