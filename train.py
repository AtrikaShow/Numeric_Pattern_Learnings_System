from pyexpat import model
import torch

from model import SimpleRegressionModel
import model
from data_generator import generate_data 

def main():
    
    # generate data
    # Let's reduce the range first to make it easier for a first model
    # Instead of 5000, let's try a smaller range for learning
    input_data, output_data = generate_data(100) 

    # Convert Python lists to PyTorch tensors
    X_tensor = torch.tensor(input_data, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(output_data, dtype=torch.float32).view(-1, 1)

    # Data Normalization
    # We calculate the Mean (average) and Standard Deviation (spread)
    X_mean, X_std = X_tensor.mean(), X_tensor.std()
    y_mean, y_std = y_tensor.mean(), y_tensor.std()

    # We shift the data so the average is 0, and spread is small (Standard Scaling)
    X_scaled = (X_tensor - X_mean) / X_std
    y_scaled = (y_tensor - y_mean) / y_std

    print(f"Original X first 3: {X_tensor[:3].flatten()}")
    print(f"Scaled X first 3:   {X_scaled[:3].flatten()}")
    
    # -------------------------------
    # 2. Initialize Model

    model = SimpleRegressionModel()
    # -------------------------------
    # ...existing code...
    loss_fn = torch.nn.MSELoss()
    
    # Let's lower the learning rate slightly to be safe
    # Learning Rate is the "step size". 0.01 is standard, but sometimes 0.001 is safer.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # -------------------------------
    # 4. Training Loop
    # -------------------------------
    epochs = 500

    for epoch in range(epochs):
        
        # USE THE SCALED DATA HERE
        predictions = model(X_scaled)   
        loss = loss_fn(predictions, y_scaled) 
        
        optimizer.zero_grad()    
        loss.backward()         
        optimizer.step()   

       
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # -------------------------------
    # 5. Test the Trained Model
    # -------------------------------
    test_values = [1.0, 5.0, 10.0]

    print("\nTesting trained model:")
    for value in test_values:
        # We must scale the test input exactly like we scaled the training input!
        test_val_tensor = torch.tensor([[value]], dtype=torch.float32)
        test_input_scaled = (test_val_tensor - X_mean) / X_std
        
        # Get prediction (this will come out scaled)
        prediction_scaled = model(test_input_scaled)
        
        # Reverse the scaling to get the real number back
        prediction_real = prediction_scaled * y_std + y_mean
        
        print(f"Input: {value} → Predicted Output: {prediction_real.item():.2f}")

    # -------------------------------
    # 6. SAVE THE TRAINED MODEL
    # -------------------------------
    checkpoint = {
        "model_state": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std
    }
    
    torch.save(checkpoint, "liner_regression_model.pth")
    print("\nSUCCESS: Model saved to liner_regression_model.pth")


if __name__ == "__main__":
    main()