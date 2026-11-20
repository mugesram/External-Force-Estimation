# External Force Estimation on a Quadrotor

Course mini project (EN4594 Autonomous Systems). An Extended Kalman Filter that estimates the external force on a
quadrotor, together with the motor thrust gain and the thrust lost to battery sag, without rotor speed sensing.

## Why

Most low cost drones have no rotor speed feedback, so the thrust is not known. As the battery drains and the motors
lag, a simple estimator mistakes the missing thrust for an external force of several newtons. This filter keeps the
thrust, the thrust gain and the motor bandwidth as states and uses the battery voltage as an input, so these effects
do not appear as a fake force.

## Results

Tested in Isaac Sim with PX4 SITL (software in the loop) and in a Python simulator, over five scenarios and four seeds.

- Vertical force RMSE 0.31 N, against 6.8 N for a linear Kalman filter and 6.7 N for a momentum observer
- 5 N force step tracked in 30 ms, 10 N impulse detected in 6 ms
- Thrust loss from battery sag estimated as 26.6 % (true 26.8 %), thrust gain within about 2 % on average
- Consistent filter in the Python simulator (force ANEES 2.1, ideal 3)

## Main code

- `efe/estimation` process and measurement models, the EKF and the baseline filters
- `efe/truth` motor, battery and external force models used by both simulators
- `efe/pysim` Python flight simulator
- `efe/io` reads the PX4 log and the ground truth into one data format
- `efe/evaluation` metrics, experiments and plots
- `sim` Isaac Sim and Pegasus launcher, `ros2_ws` ROS 2 offboard node
- `report` the project report and the script that makes its figures
