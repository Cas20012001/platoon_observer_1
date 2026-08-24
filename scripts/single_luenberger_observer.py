#!/usr/bin/env python3

import numpy as np
import rospy

from std_msgs.msg import Float32, Float32MultiArray


class SingleLuenbergerObserver:
    def __init__(self):
        self.Ts = rospy.get_param("~Ts", 0.1)

        error_topic = rospy.get_param(
            "~error_topic",
            "/spacing_error_meas_2"
        )

        velocity_topic = rospy.get_param(
            "~velocity_topic",
            "/vicon_velocity_meas_2"
        )

        leader_acc_topic = rospy.get_param(
            "~leader_acc_topic",
            "/acc_saturated_1"
        )

        follower_acc_topic = rospy.get_param(
            "~follower_acc_topic",
            "/acc_saturated_2"
        )

        state_topic = rospy.get_param(
            "~state_topic",
            "/state_estimate_2"
        )

        # State:
        # x = [e, v2, a2, delta_v, a1]^T

        self.A = np.array([
            [1.0, 0.0, -0.0352848, 0.1, 0.0036788],
            [0.0, 1.0,  0.0632121, 0.0, 0.0],
            [0.0, 0.0,  0.3678794, 0.0, 0.0],
            [0.0, 0.0, -0.0632121, 1.0, 0.0632121],
            [0.0, 0.0,  0.0, 0.0, 0.3678794]
        ])

        # Leader input u1.
        self.B1 = np.array([
            0.0013212,
            0.0,
            0.0,
            0.0367879,
            0.6321206
        ])

        # Follower input u2.
        self.B2 = np.array([
            -0.0197152,
             0.0367879,
             0.6321206,
            -0.0367879,
             0.0
        ])

        self.C = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0]
        ])

        # Gain from your MATLAB pole placement.
        self.L = np.array([
            [ 1.1322, -0.0253],
            [-0.0054,  0.6035],
            [ 0.0067, -0.0698],
            [ 3.1489, -0.0805],
            [ 0.0187,  0.0547]
        ])

        self.x_hat = np.zeros(5)

        self.error_measured = 0.0
        self.velocity_measured = 0.0
        self.u1 = 0.0
        self.u2 = 0.0

        self.have_error = False
        self.have_velocity = False
        self.initialized = False

        self.state_pub = rospy.Publisher(
            state_topic,
            Float32MultiArray,
            queue_size=1
        )

        rospy.Subscriber(
            error_topic,
            Float32,
            self.error_callback,
            queue_size=1
        )

        rospy.Subscriber(
            velocity_topic,
            Float32,
            self.velocity_callback,
            queue_size=1
        )

        rospy.Subscriber(
            leader_acc_topic,
            Float32,
            self.leader_acc_callback,
            queue_size=1
        )

        rospy.Subscriber(
            follower_acc_topic,
            Float32,
            self.follower_acc_callback,
            queue_size=1
        )

        rospy.Timer(
            rospy.Duration(self.Ts),
            self.update
        )

    def error_callback(self, msg):
        self.error_measured = msg.data
        self.have_error = True

    def velocity_callback(self, msg):
        self.velocity_measured = msg.data
        self.have_velocity = True

    def leader_acc_callback(self, msg):
        self.u1 = msg.data

    def follower_acc_callback(self, msg):
        self.u2 = msg.data

    def update(self, event):
        if not self.have_error or not self.have_velocity:
            return

        y = np.array([
            self.error_measured,
            self.velocity_measured
        ])

        if not self.initialized:
            self.x_hat[0] = self.error_measured
            self.x_hat[1] = self.velocity_measured
            self.initialized = True
        else:
            residual = y - self.C @ self.x_hat

            self.x_hat = (
                self.A @ self.x_hat
                + self.B1 * self.u1
                + self.B2 * self.u2
                + self.L @ residual
            )

        self.state_pub.publish(
            Float32MultiArray(
                data=self.x_hat.tolist()
            )
        )


if __name__ == "__main__":
    rospy.init_node("single_luenberger_observer")
    SingleLuenbergerObserver()
    rospy.spin()
