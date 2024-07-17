extends Area2D

var speed = 12

func _physics_process(_delta):
	global_position.x -= speed

func _on_body_entered(body):
	if body is CharacterBody2D:
		body.colliding = true
		body.reward = -100
		body.game_end = true
		call_deferred("queue_free")
