limit = 50

def generate_data(limit):
    input_data = []
    output_data = []
    for i in range(limit):
        input_value = i
        output_value = 3* input_value + 2 

        input_data.append(input_value)
        output_data.append(output_value)

    return input_data, output_data

input_data, output_data = generate_data(limit)

#print("Input Data:", input_data[:40])
#print("Output Data:", output_data[:40])

# export this function for use in other modules
