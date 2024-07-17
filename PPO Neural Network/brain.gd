extends Node
class_name Brain

var policy_network : PolicyValueNetwork = null
var value_network : PolicyValueNetwork = null
var host = null

var trajectories = []
var gamma = 0.99

func create(input_size,hidden_layer_size,hidden_layer_amount,output_size,new_host):
	host = new_host
	policy_network = PolicyValueNetwork.new(input_size,hidden_layer_size,hidden_layer_amount,output_size,"Policy")
	value_network = PolicyValueNetwork.new(input_size,hidden_layer_size,hidden_layer_amount,output_size,"Value")
	return self

func get_action(states) -> Array:
	return policy_network.forward(states)

# In host, collect each of these in one frame. States should always come first
func collect(states=null,action=null,reward=null,next_states=null):
	var trajectory_buffer = [states,action,reward,next_states]
	if states != null:
		trajectory_buffer[0] = states
	if action != null:
		trajectory_buffer[1] = action
	if reward != null:
		trajectory_buffer[2] = reward
	if next_states != null:
		trajectory_buffer[3] = next_states
	trajectories.append({
		"states": trajectory_buffer[0],
		"action": trajectory_buffer[1],
		"reward": trajectory_buffer[2],
		"next_states": trajectory_buffer[3]
	})
	return

func compute_log_probs(action_probs,action):
	return log(action_probs[action])

func compute_policy_loss_gradient(log_probs: float, advantage: float, old_log_probs: float):
	var ratio = exp(log_probs - old_log_probs)
	var clip_value = 2.0
	var clipped_ratio = clamp(ratio, 1.0 - clip_value, 1.0 + clip_value)
	var loss_gradient = -min(ratio * advantage, clipped_ratio * advantage)
	return loss_gradient

func choose_action(action_probs):
	var cumulative_prob = 0.0
	var random_value = randf()
	for i in range(action_probs.size()):
		cumulative_prob += action_probs[i]
		if random_value < cumulative_prob:
			return i
	return action_probs.size() - 1

func update_policy_network(advantage):
	for transition in trajectories:
		var states = transition["states"]
		var action_probs = policy_network.forward(states)
		var action = choose_action(action_probs)
		var old_log_probs = log(action_probs[action])
		var next_states = transition["next_states"]
		var next_action_probs = policy_network.forward(next_states)
		var next_action = choose_action(next_action_probs)
		var log_probs = log(next_action_probs[next_action])
		var policy_gradient = compute_policy_loss_gradient(log_probs, advantage, old_log_probs)
		policy_network.update_gradients(states,policy_gradient)

func update_value_network():
	for transition in trajectories:
		var states = transition["states"]
		var next_states = transition["next_states"]
		var actual_reward = transition["reward"]
		var estimated_reward = value_network.forward(states)
		var estimated_future_reward = value_network.forward(next_states)
		var td_error = actual_reward + (gamma * estimated_future_reward) - estimated_reward
		value_network.update_gradients(states, td_error)
		return td_error

func get_neuron_weights_and_biases():
	return null

func end():
	var advantage = update_value_network()
	update_policy_network(advantage)
	trajectories.clear()
