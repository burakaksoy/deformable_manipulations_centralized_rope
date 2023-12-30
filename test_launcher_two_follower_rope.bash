#!/bin/bash
sleep 1s;

gnome-terminal --tab --title="ROSCORE" --command "bash -c \"source ~/.bashrc; killall gzclient && killall gzserver; roscore; exec bash\"";
sleep 1s;

gnome-terminal --tab --title="All" --command "bash -c \"source ~/.bashrc; roslaunch deformable_manipulations_centralized_rope two_follower_rope.launch launch_controller:=false; exec bash\"";
sleep 4s;

# Aux enabled
# gnome-terminal --tab --title="Controller" --command "bash -c \"source ~/.bashrc; roslaunch deformable_manipulations_centralized_rope dlo_velocity_controller_2d_safe.launch aux_enabled:=False aux_bias_direction:='up'; exec bash\"";
# Aux disabled
# gnome-terminal --tab --title="Controller" --command "bash -c \"source ~/.bashrc; roslaunch deformable_manipulations_centralized_rope dlo_velocity_controller_2d_safe.launch; exec bash\"";
sleep 1s;







