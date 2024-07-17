extends CharacterBody2D

@export var jump_height = -1500.0
var brain : Brain
var reward : int = 0
var game_end = false

# Get the gravity from the project settings to be synced with RigidBody nodes.
var gravity = 100
var is_jumping : bool = false; var distance_to_object : float = 1024.0
var action = "idle"; var end = false; var colliding = false; var last_probs

func _ready():
	if brain == null:
		brain = get_parent().create(5,24,3,2,self)
	await get_tree().process_frame

func _physics_process(_delta):
	reward += 1
	get_tree().root.get_child(0).reward = reward; var object_speed = 0.0
	if get_parent().get_node("RayCast2D").get_collider():
		var closest_obstacle = get_parent().get_node("RayCast2D").get_collider()
		distance_to_object = closest_obstacle.global_position.x - global_position.x
		object_speed = closest_obstacle.speed
	else:
		distance_to_object = 1024
	
	var current_states = [distance_to_object,int(is_on_floor()),object_speed,global_position.y,int(colliding)]
	if is_on_floor():
		is_jumping = false
	else:
		velocity.y += gravity
	var action_probs = brain.get_action(current_states)
	if action_probs != last_probs:
		print(action_probs)
	last_probs = action_probs
	var random_float = randf_range(0.0,1.0)
	if random_float <= action_probs[0]:
		action = "idle"
		reward += 1
	else:
		if is_on_floor():
			action = "jump"
			velocity.y = jump_height
			if distance_to_object > 256: reward -= 5
			else: reward += 2
	var next_states = [distance_to_object,int(is_on_floor()),object_speed,global_position.y,int(colliding)]
	if game_end == true:
		wrap_up()
	brain.collect(current_states,action,reward,next_states)
	move_and_slide()

func wrap_up():
	brain.end()
	get_parent().get_parent().new_run()
	global_position = get_parent().get_parent().get_node("Marker2D").global_position
	reward = 0; game_end = false
