import torch  # core tensor library
import torch.nn as nn  # neural network module 

# define the model class
class SimpleRegressionModel(nn.Module):
       def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)

       def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x) # ReLU adds non-linearity, so the model can learn more complex patterns.
        x = self.layer2(x)
        return x
       

# this will only run if this file is executed directly not imported
if __name__ == "__main__":
        # instantiate the model
        model = SimpleRegressionModel()
        # print the model architecture
        print(model)

        # test with dummy input
        test_input = torch.tensor([[1.0]])
        test_output = model(test_input)
        print("Test output for input 5.0:", test_output)