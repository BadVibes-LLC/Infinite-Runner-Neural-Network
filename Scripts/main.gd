extends Node2D

var ai_brain = null
var AI = preload("res://Scenes/ai.tscn")
var Obstacle = preload("res://Scenes/obstacle.tscn")
var ob_array = []
var run = 0
var reward = 0
@onready var ai = $Brain/AI
@onready var label = $CanvasLayer/Label

func _physics_process(_delta):
	label.set_text("Current Iteration: %s\nCurrent Reward: %s" % [run,reward])

func new_run():
	run += 1
	for y in $Obstacles.get_children():
		y.queue_free()

func _ready():
	randomize()
	$Timer.start(randf_range(0.0,1.0))

func _on_timer_timeout():
	var ob = Obstacle.instantiate()
	ob.speed = 12.0
	ob.global_position = Vector2(1240,490)
	$Obstacles.add_child(ob)
	$Timer.start(randf_range(0.5,1.5))

func _on_area_2d_area_entered(area):
	area.queue_free()
