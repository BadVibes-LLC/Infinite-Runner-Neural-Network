extends Node
class_name PolicyValueNetwork

var layers = []
var output_layer = []
var type = null

func _init(input_size,hidden_layer_size,hidden_layer_amount,output_size,new_type):
	randomize()
	type = new_type
	if type == "Value":
		output_size = 1
	for x in range(hidden_layer_amount):
		var hidden_layer = []
		var previous_layer_size
		if x == 0:
			previous_layer_size = input_size 
		else:
			previous_layer_size = hidden_layer_size
		for y in range(hidden_layer_size):
			var neuron = Neuron.new("hidden_layer",previous_layer_size)
			hidden_layer.append(neuron)
		layers.append(hidden_layer)
	
	var previous_layer_size = hidden_layer_size if hidden_layer_amount > 0 else input_size
	for x in range(output_size):
		var neuron = Neuron.new("output_layer",previous_layer_size)
		output_layer.append(neuron)

func forward(input_array : PackedFloat32Array):
	var current_input = input_array
	for hidden_layer in layers:
		var hidden_outputs : PackedFloat32Array = []
		for neuron in hidden_layer:
			hidden_outputs.append(neuron.process(current_input))
		current_input = hidden_outputs
	
	var raw_outputs : PackedFloat32Array = []
	for neuron in output_layer:
		raw_outputs.append(neuron.process(current_input))
	if type == "Policy":
		return softmax(raw_outputs)
	elif type == "Value":
		return raw_outputs[0]
	else:
		print("Incorrect Type")
		return null

func update_gradients(input_values,gradient):
	for hidden_layer in layers:
		var input = []
		for neuron in hidden_layer:
			neuron.compute_gradients(input_values,gradient)
			input.append(neuron.process(input_values))
		input_values = input
	for neuron in output_layer:
		neuron.compute_gradients(input_values,gradient)

func softmax(inputs: PackedFloat32Array):
	var max_inputs = -INF
	for i in range(inputs.size()):
		if inputs[i] > max_inputs:
			max_inputs = inputs[i]

	var exp_sum = 0.0
	var exp_values : PackedFloat32Array = []
	for i in range(inputs.size()):
		var exp_value = exp(inputs[i] - max_inputs)
		exp_values.append(exp_value)
		exp_sum += exp_value

	var softmax_values : PackedFloat32Array = []
	for i in range(exp_values.size()):
		softmax_values.append(exp_values[i] / exp_sum)
	
	return softmax_values
