extends Node
class_name Neuron

var weights : PackedFloat32Array = []
var bias : float = 0.0
var type = null
var alpha = 0.01

func _init(new_type,num_inputs):
	randomize()
	type = new_type
	initialize_weights(num_inputs)

func initialize_weights(num_inputs):
	var stddev = sqrt(2.0 / float(num_inputs))
	for i in range(num_inputs):
		weights.append(randf() * 2.0 * stddev - stddev)
	bias = randf() * 2.0 * stddev - stddev

func process(inputs : Array) -> float:
	var output = activation_function(inputs)
	if type == "hidden_layer":
		return relu(output)
	return output

func activation_function(inputs) -> float:
	var output = 0.0
	for i in range(inputs.size()):
		output += float(inputs[i]) * weights[i]
	output += bias
	return output

func relu(x) -> float:
	var clip = 1.0
	return min(max(0, x),clip)

func compute_gradients(input_values,loss_gradient):
	for i in range(weights.size()):
		weights[i] -= alpha * (input_values[i] * loss_gradient)
	bias -= alpha * loss_gradient
