import torch  # core tensor library
import torch.nn as nn  # neural network module 

# define the model class
class SimpleRegressionmodel(nn.Module):
       def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)

       def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x) # ReLU adds non-linearity, so the model can learn more complex patterns.
        x = self.layer2(x)
        return x
       


# instantiate the model
model = SimpleRegressionmodel()
# print the model architecture
print(model)

# test with dummy input
test_input = torch.tensor([[5.0]])
test_output = model(test_input)
print("Test output for input 5.0:", test_output)