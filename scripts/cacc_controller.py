#!/usr/bin/env python3

import math
import numpy as np
import rospy

from std_msgs.msg import Float32, Float32MultiArray


class CACCController:
    def __init__(self):
        self.Ts = rospy.get_param("~Ts", 0.1)
        self.h = rospy.get_param("~h", 0.5)
        self.tau = rospy.get_param("~tau", 0.1)

        # Initial estimated gains.
        self.kp = rospy.get_param("~kp", 0.4)
        self.kd = rospy.get_param("~kd", 0.8)
        self.kdd = rospy.get_param("~kdd", 0.1)

        self.min_acc = rospy.get_param("~min_acc", -0.5)
        self.max_acc = rospy.get_param("~max_acc", 0.5)

        self.enabled = rospy.get_param("~enabled", False)

        state_topic = rospy.get_param(
            "~state_topic",
            "/state_estimate_2"
        )

        leader_acc_topic = rospy.get_param(
            "~leader_acc_topic",
            "/acc_saturated_1"
        )

        follower_acc_topic = rospy.get_param(
            "~follower_acc_topic",
            "/acc_saturated_2"
        )

        desired_acc_topic = rospy.get_param(
            "~desired_acc_topic",
            "/desired_acc_2"
        )

        self.x_hat = np.zeros(5)
        self.have_state = False

        self.u1 = 0.0
        self.u2_applied = 0.0
        self.last_command = 0.0
        self.have_u2_feedback = False

        self.desired_acc_pub = rospy.Publisher(
            desired_acc_topic,
            Float32,
            queue_size=1
        )

        rospy.Subscriber(
            state_topic,
            Float32MultiArray,
            self.state_callback,
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

    def state_callback(self, msg):
        if len(msg.data) >= 5:
            self.x_hat = np.array(msg.data[:5])
            self.have_state = True

    def leader_acc_callback(self, msg):
        self.u1 = msg.data

    def follower_acc_callback(self, msg):
        self.u2_applied = msg.data
        self.have_u2_feedback = True

    def update(self, event):
        if not self.enabled or not self.have_state:
            return

        e_hat = self.x_hat[0]
        a2_hat = self.x_hat[2]
        delta_v_hat = self.x_hat[3]
        a1_hat = self.x_hat[4]

        if self.have_u2_feedback:
            u2 = self.u2_applied
        else:
            u2 = self.last_command

        # First derivative of spacing error:
        # e_dot = delta_v - h*a2
        e_dot_hat = delta_v_hat - self.h * a2_hat

        # Second derivative of spacing error, as in the paper:
        e_ddot_hat = (
            a1_hat
            + (self.h / self.tau - 1.0) * a2_hat
            - (self.h / self.tau) * u2
        )

        xi = (
            self.kp * e_hat
            + self.kd * e_dot_hat
            + self.kdd * e_ddot_hat
            + self.u1
        )

        alpha = math.exp(-self.Ts / self.h)

        u_next = (
            alpha * u2
            + (1.0 - alpha) * xi
        )

        u_next = float(np.clip(
            u_next,
            self.min_acc,
            self.max_acc
        ))

        self.last_command = u_next

        self.desired_acc_pub.publish(
            Float32(data=u_next)
        )


if __name__ == "__main__":
    rospy.init_node("cacc_controller")
    CACCController()
    rospy.spin()
