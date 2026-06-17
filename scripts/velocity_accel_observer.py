#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import Imu


class VelocityAccelerationObserver:
    def __init__(self):
        rospy.init_node("velocity_accel_observer")

        # -----------------------------
        # Topics
        # -----------------------------
        self.wheel_velocity_topic = rospy.get_param("~wheel_velocity_topic", "/wheel_velocity")
        self.imu_topic = rospy.get_param("~imu_topic", "/imu")
        self.accel_cmd_topic = rospy.get_param("~accel_cmd_topic", "/accel_cmd")
        self.output_topic = rospy.get_param("~output_topic", "/state_estimate")

        # -----------------------------
        # Observer timing
        # These matrices are valid for Ts = 0.1 s
        # -----------------------------
        self.Ts = rospy.get_param("~sample_time", 0.1)

        # -----------------------------
        # Discrete system matrices
        # x = [v, a]^T
        # -----------------------------
        self.Ad_11 = rospy.get_param("~Ad_11", 1.0000)
        self.Ad_12 = rospy.get_param("~Ad_12", 0.0632)
        self.Ad_21 = rospy.get_param("~Ad_21", 0.0)
        self.Ad_22 = rospy.get_param("~Ad_22", 0.3679)

        self.Bd_1 = rospy.get_param("~Bd_1", 0.0368)
        self.Bd_2 = rospy.get_param("~Bd_2", 0.6321)

        # Luenberger observer gain
        self.Ld_11 = rospy.get_param("~Ld_11", 0.6321)
        self.Ld_12 = rospy.get_param("~Ld_12", 0.0632)
        self.Ld_21 = rospy.get_param("~Ld_21", 0.0)
        self.Ld_22 = rospy.get_param("~Ld_22", 0.2325)

        # -----------------------------
        # Sensor conventions
        # -----------------------------
        self.velocity_sign = rospy.get_param("~velocity_sign", 1.0)
        self.accel_sign = rospy.get_param("~accel_sign", 1.0)
        self.command_sign = rospy.get_param("~command_sign", 1.0)

        self.imu_accel_axis = rospy.get_param("~imu_accel_axis", "x")
        self.imu_accel_bias = rospy.get_param("~imu_accel_bias", 0.0)

        # -----------------------------
        # State estimate
        # -----------------------------
        self.v_hat = rospy.get_param("~initial_velocity", 0.0)
        self.a_hat = rospy.get_param("~initial_acceleration", 0.0)

        # Latest measurements
        self.v_meas = None
        self.a_meas = None
        self.u_cmd = 0.0

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        rospy.Subscriber(self.wheel_velocity_topic, Float64, self.wheel_velocity_callback)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        rospy.Subscriber(self.accel_cmd_topic, Float64, self.accel_cmd_callback)

        self.state_pub = rospy.Publisher(self.output_topic, Float64MultiArray, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(self.Ts), self.update_observer)

        rospy.loginfo("Velocity-acceleration observer started.")
        rospy.loginfo("Publishing state estimate to %s", self.output_topic)

    def wheel_velocity_callback(self, msg):
        self.v_meas = self.velocity_sign * msg.data

    def imu_callback(self, msg):
        if self.imu_accel_axis == "x":
            raw_accel = msg.linear_acceleration.x
        elif self.imu_accel_axis == "y":
            raw_accel = msg.linear_acceleration.y
        elif self.imu_accel_axis == "z":
            raw_accel = msg.linear_acceleration.z
        else:
            rospy.logwarn_throttle(2.0, "Invalid imu_accel_axis. Use x, y, or z.")
            raw_accel = msg.linear_acceleration.x

        self.a_meas = self.accel_sign * (raw_accel - self.imu_accel_bias)

    def accel_cmd_callback(self, msg):
        self.u_cmd = self.command_sign * msg.data

    def update_observer(self, event):
        if self.v_meas is None:
            rospy.logwarn_throttle(2.0, "Waiting for wheel velocity measurement.")
            return

        if self.a_meas is None:
            rospy.logwarn_throttle(2.0, "Waiting for IMU acceleration measurement.")
            return

        # Current estimate
        v_hat_k = self.v_hat
        a_hat_k = self.a_hat

        # Measurements y = [v_meas, a_meas]^T
        v_error = self.v_meas - v_hat_k
        a_error = self.a_meas - a_hat_k

        # Model prediction:
        #
        # x_pred = Ad*x_hat + Bd*u
        #
        v_pred = (
            self.Ad_11 * v_hat_k
            + self.Ad_12 * a_hat_k
            + self.Bd_1 * self.u_cmd
        )

        a_pred = (
            self.Ad_21 * v_hat_k
            + self.Ad_22 * a_hat_k
            + self.Bd_2 * self.u_cmd
        )

        # Observer correction:
        #
        # x_hat_next = x_pred + Ld*(y - C*x_hat)
        # C = I, so y - C*x_hat = [v_error, a_error]
        #
        self.v_hat = (
            v_pred
            + self.Ld_11 * v_error
            + self.Ld_12 * a_error
        )

        self.a_hat = (
            a_pred
            + self.Ld_21 * v_error
            + self.Ld_22 * a_error
        )

        # Publish [v_hat, a_hat]
        msg = Float64MultiArray()
        msg.data = [self.v_hat, self.a_hat]
        self.state_pub.publish(msg)


if __name__ == "__main__":
    try:
        VelocityAccelerationObserver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
