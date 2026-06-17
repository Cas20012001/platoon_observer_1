# platoon_observer

ROS Noetic package for a velocity-acceleration Luenberger observer.

## Node

`velocity_accel_observer.py`

Subscribes to:

- `/wheel_velocity` (`std_msgs/Float64`)
- `/imu` (`sensor_msgs/Imu`)
- `/accel_cmd` (`std_msgs/Float64`)

Publishes:

- `/state_estimate` (`std_msgs/Float64MultiArray`)

where:

- `data[0] = v_hat`
- `data[1] = a_hat`

## Run

```bash
roslaunch platoon_observer velocity_accel_observer.launch
