#!/usr/bin/env python
from __future__ import division

import math
import threading

import rospy
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped


class ViconDerivativeObserver(object):
    def __init__(self):
        self.lock = threading.Lock()

        self.car_number = int(
            rospy.get_param("~car_number", 3)
        )

        self.vicon_topic = rospy.get_param(
            "~vicon_topic",
            "/vicon/jetracer{}".format(self.car_number)
        )

        # Only used to read throttle for the model input u.
        # Encoder velocity and IMU acceleration are NOT used as measurements.
        self.sensor_topic = rospy.get_param(
            "~sensor_topic",
            "/sensors_and_input_{}".format(self.car_number)
        )

        self.state_estimate_topic = rospy.get_param(
            "~state_estimate_topic",
            "/vicon_state_estimate_{}".format(self.car_number)
        )

        self.observer_input_acc_topic = rospy.get_param(
            "~observer_input_acc_topic",
            "/vicon_observer_input_acc_{}".format(self.car_number)
        )

        self.vicon_velocity_topic = rospy.get_param(
            "~vicon_velocity_meas_topic",
            "/vicon_velocity_meas_{}".format(self.car_number)
        )

        self.vicon_acceleration_topic = rospy.get_param(
            "~vicon_acceleration_meas_topic",
            "/vicon_acceleration_meas_{}".format(self.car_number)
        )

        self.control_rate = float(
            rospy.get_param("~control_rate", 10.0)
        )

        self.dt_nominal = 1.0 / self.control_rate

        # Vicon derivative filtering
        self.vicon_timeout = float(
            rospy.get_param("~vicon_timeout", 0.30)
        )

        self.min_vicon_dt = float(
            rospy.get_param("~min_vicon_dt", 0.005)
        )

        self.max_vicon_dt = float(
            rospy.get_param("~max_vicon_dt", 0.50)
        )

        self.max_reasonable_speed = float(
            rospy.get_param("~max_reasonable_speed", 5.0)
        )

        self.max_reasonable_acc = float(
            rospy.get_param("~max_reasonable_acc", 8.0)
        )

        # Smaller alpha = smoother but more delay
        self.velocity_alpha = float(
            rospy.get_param("~velocity_alpha", 0.20)
        )

        self.acceleration_alpha = float(
            rospy.get_param("~acceleration_alpha", 0.08)
        )

        # Observer model:
        # x = [v, a]
        # v_dot = a
        # a_dot = -1/tau * a + 1/tau * u
        self.observer_tau = float(
            rospy.get_param("~observer_tau", 0.1)
        )

        self.update_discrete_model(self.dt_nominal)

        # Original observer gain
        self.L00 = float(rospy.get_param("~L00", 0.6321))
        self.L01 = float(rospy.get_param("~L01", 0.0632))
        self.L10 = float(rospy.get_param("~L10", 0.0))
        self.L11 = float(rospy.get_param("~L11", 0.2325))

        # DART throttle-to-acceleration model parameters.
        # This is only for the observer input u.
        self.use_throttle_model = bool(
            rospy.get_param("~use_throttle_model", True)
        )

        self.mass = float(rospy.get_param("~mass", 1.67))

        self.a_m = float(
            rospy.get_param("~a_m", 28.887779235839844)
        )

        self.b_m = float(
            rospy.get_param("~b_m", 5.986172199249268)
        )

        self.c_m = float(
            rospy.get_param("~c_m", -0.15045104920864105)
        )

        self.a_f = float(
            rospy.get_param("~a_f", 1.7194761037826538)
        )

        self.b_f = float(
            rospy.get_param("~b_f", 13.312559127807617)
        )

        self.c_f = float(
            rospy.get_param("~c_f", 0.289848655462265)
        )

        # Internal Vicon state
        self.have_vicon_position = False
        self.have_vicon_velocity = False
        self.have_vicon_acceleration = False

        self.last_vicon_receive_time = None
        self.last_vicon_stamp = None

        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0

        self.raw_vicon_speed = 0.0
        self.vicon_speed = 0.0

        self.raw_vicon_acceleration = 0.0
        self.vicon_acceleration = 0.0

        # Input model state
        self.throttle = 0.0
        self.u_input_acc = 0.0
        self.have_sensor_data = False

        # Observer state
        self.observer_initialized = False
        self.v_hat = 0.0
        self.a_hat = 0.0

        # Publishers
        self.state_publisher = rospy.Publisher(
            self.state_estimate_topic,
            Float32MultiArray,
            queue_size=1
        )

        self.observer_input_acc_publisher = rospy.Publisher(
            self.observer_input_acc_topic,
            Float32,
            queue_size=1
        )

        self.vicon_velocity_publisher = rospy.Publisher(
            self.vicon_velocity_topic,
            Float32,
            queue_size=1
        )

        self.vicon_acceleration_publisher = rospy.Publisher(
            self.vicon_acceleration_topic,
            Float32,
            queue_size=1
        )

        # Subscribers
        rospy.Subscriber(
            self.vicon_topic,
            PoseWithCovarianceStamped,
            self.vicon_callback,
            queue_size=1
        )

        rospy.Subscriber(
            self.sensor_topic,
            Float32MultiArray,
            self.sensor_callback,
            queue_size=1
        )

        rospy.loginfo(
            "Vicon derivative observer started for car %d",
            self.car_number
        )

        rospy.loginfo(
            "Vicon topic: %s",
            self.vicon_topic
        )

        rospy.loginfo(
            "Publishing estimate to: %s",
            self.state_estimate_topic
        )

        rospy.logwarn(
            "This observer does NOT use encoder velocity or IMU acceleration as measurements."
        )

    def update_discrete_model(self, dt):
        tau = self.observer_tau
        exp_term = math.exp(-dt / tau)

        self.Ad00 = 1.0
        self.Ad01 = tau * (1.0 - exp_term)
        self.Ad10 = 0.0
        self.Ad11 = exp_term

        self.Bd0 = dt - tau * (1.0 - exp_term)
        self.Bd1 = 1.0 - exp_term

    def vicon_callback(self, msg):
        now_receive = rospy.Time.now()

        stamp = msg.header.stamp
        if stamp.to_sec() <= 0.0:
            stamp = now_receive

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        z = float(msg.pose.pose.position.z)

        with self.lock:
            if self.have_vicon_position:
                dt = (stamp - self.last_vicon_stamp).to_sec()

                if dt > self.min_vicon_dt and dt < self.max_vicon_dt:
                    dx = x - self.last_x
                    dy = y - self.last_y

                    distance_xy = math.sqrt(dx * dx + dy * dy)
                    raw_speed = distance_xy / dt

                    if raw_speed <= self.max_reasonable_speed:
                        self.raw_vicon_speed = raw_speed

                        if not self.have_vicon_velocity:
                            self.vicon_speed = raw_speed
                            self.have_vicon_velocity = True
                        else:
                            previous_speed = self.vicon_speed

                            alpha_v = self.velocity_alpha
                            self.vicon_speed = (
                                alpha_v * raw_speed
                                + (1.0 - alpha_v) * self.vicon_speed
                            )

                            raw_acc = (
                                self.vicon_speed - previous_speed
                            ) / dt

                            if abs(raw_acc) <= self.max_reasonable_acc:
                                self.raw_vicon_acceleration = raw_acc

                                if not self.have_vicon_acceleration:
                                    self.vicon_acceleration = raw_acc
                                    self.have_vicon_acceleration = True
                                else:
                                    alpha_a = self.acceleration_alpha
                                    self.vicon_acceleration = (
                                        alpha_a * raw_acc
                                        + (1.0 - alpha_a)
                                        * self.vicon_acceleration
                                    )

            self.last_x = x
            self.last_y = y
            self.last_z = z
            self.last_vicon_stamp = stamp
            self.last_vicon_receive_time = now_receive
            self.have_vicon_position = True

    def sensor_callback(self, msg):
        # Confirmed layout:
        # data[8] = throttle
        if len(msg.data) <= 8:
            return

        throttle = float(msg.data[8])

        with self.lock:
            self.throttle = throttle
            self.have_sensor_data = True

            if self.use_throttle_model:
                self.u_input_acc = self.acceleration_from_throttle(
                    max(0.0, self.vicon_speed),
                    throttle
                )
            else:
                self.u_input_acc = 0.0

    def friction_force(self, velocity):
        return (
            -self.a_f * math.tanh(self.b_f * velocity)
            -self.c_f * velocity
        )

    def acceleration_from_throttle(self, velocity, throttle):
        w = 0.5 * (
            math.tanh(100.0 * (throttle + self.c_m))
            + 1.0
        )

        motor_force = (
            (self.a_m - velocity * self.b_m)
            * w
            * (throttle + self.c_m)
        )

        friction_force = self.friction_force(velocity)

        return (motor_force + friction_force) / self.mass

    def vicon_is_valid(self, now):
        if not self.have_vicon_velocity:
            rospy.logwarn_throttle(
                1.0,
                "Waiting for Vicon-derived velocity"
            )
            return False

        if not self.have_vicon_acceleration:
            rospy.logwarn_throttle(
                1.0,
                "Waiting for Vicon-derived acceleration"
            )
            return False

        if self.last_vicon_receive_time is None:
            return False

        age = (now - self.last_vicon_receive_time).to_sec()

        if age > self.vicon_timeout:
            rospy.logerr_throttle(
                1.0,
                "Vicon timeout: %.3f s",
                age
            )
            return False

        return True

    def update_observer(self, v_meas, a_meas, u_acc):
        if not self.observer_initialized:
            self.v_hat = v_meas
            self.a_hat = a_meas
            self.observer_initialized = True
            return

        # Prediction
        v_pred = (
            self.Ad00 * self.v_hat
            + self.Ad01 * self.a_hat
            + self.Bd0 * u_acc
        )

        a_pred = (
            self.Ad11 * self.a_hat
            + self.Bd1 * u_acc
        )

        # Measurement residual
        error_v = v_meas - v_pred
        error_a = a_meas - a_pred

        # Correction
        self.v_hat = (
            v_pred
            + self.L00 * error_v
            + self.L01 * error_a
        )

        self.a_hat = (
            a_pred
            + self.L10 * error_v
            + self.L11 * error_a
        )

    def publish_state(self, v_meas, a_meas, u_acc):
        msg = Float32MultiArray()

        msg.data = [
            self.v_hat,                 # data[0]
            self.a_hat,                 # data[1]
            v_meas,                     # data[2]
            a_meas,                     # data[3]
            u_acc,                      # data[4]
            self.throttle,              # data[5]
            self.raw_vicon_speed,       # data[6]
            self.raw_vicon_acceleration # data[7]
        ]

        self.state_publisher.publish(msg)

        self.observer_input_acc_publisher.publish(
            Float32(data=u_acc)
        )

        self.vicon_velocity_publisher.publish(
            Float32(data=v_meas)
        )

        self.vicon_acceleration_publisher.publish(
            Float32(data=a_meas)
        )

    def control_step(self):
        now = rospy.Time.now()

        with self.lock:
            valid = self.vicon_is_valid(now)

            v_meas = self.vicon_speed
            a_meas = self.vicon_acceleration
            u_acc = self.u_input_acc

        if not valid:
            return

        self.update_observer(
            v_meas,
            a_meas,
            u_acc
        )

        self.publish_state(
            v_meas,
            a_meas,
            u_acc
        )

    def run(self):
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown():
            self.control_step()
            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node(
            "vicon_derivative_observer",
            anonymous=False
        )

        observer = ViconDerivativeObserver()
        observer.run()

    except rospy.ROSInterruptException:
        pass
