#!/usr/bin/env python3

import os

import numpy as np
import rospy
from std_msgs.msg import Float32, Float32MultiArray


class LeaderVelocityController:
    def __init__(self):
        car_number = rospy.get_param(
            "~car_number",
            os.environ.get("car_number", "1")
        )

        self.car_number = str(car_number)

        self.kp = float(rospy.get_param("~kp", 1.0)) # was 0.5
        self.min_acc = float(rospy.get_param("~min_acc", -1.0)) # was -0.5
        self.max_acc = float(rospy.get_param("~max_acc", 1.0)) # was 0.5

        self.velocity_index = int(
            rospy.get_param("~velocity_index", 6)
        )

        self.v_ref = 0.0
        self.v = 0.0
        self.velocity_received = False

        sensors_topic = rospy.get_param(
            "~sensors_topic",
            "/sensors_and_input_" + self.car_number
        )

        v_ref_topic = rospy.get_param(
            "~v_ref_topic",
            "/v_ref_" + self.car_number
        )

        desired_acc_topic = rospy.get_param(
            "~desired_acc_topic",
            "/desired_acc_" + self.car_number
        )

        self.desired_acc_pub = rospy.Publisher(
            desired_acc_topic,
            Float32,
            queue_size=1
        )

        rospy.Subscriber(
            sensors_topic,
            Float32MultiArray,
            self.sensors_callback,
            queue_size=1
        )

        rospy.Subscriber(
            v_ref_topic,
            Float32,
            self.v_ref_callback,
            queue_size=1
        )

        rospy.loginfo(
            "Leader velocity controller started for car %s",
            self.car_number
        )
        rospy.loginfo("Sensors topic: %s", sensors_topic)
        rospy.loginfo("Velocity reference topic: %s", v_ref_topic)
        rospy.loginfo("Desired acceleration topic: %s", desired_acc_topic)

    def sensors_callback(self, msg):
        data = np.asarray(msg.data, dtype=float)

        if len(data) <= self.velocity_index:
            rospy.logwarn_throttle(
                2.0,
                "sensors_and_input message has %d elements, "
                "but velocity_index is %d",
                len(data),
                self.velocity_index
            )
            return

        self.v = float(data[self.velocity_index])
        self.velocity_received = True

    def v_ref_callback(self, msg):
        self.v_ref = float(msg.data)

    def update(self):
        if not self.velocity_received:
            rospy.logwarn_throttle(
                2.0,
                "Waiting for velocity measurement"
            )
            return

        velocity_error = self.v_ref - self.v

        desired_acc = self.kp * velocity_error

        desired_acc = np.clip(
            desired_acc,
            self.min_acc,
            self.max_acc
        )

        self.desired_acc_pub.publish(
            Float32(data=float(desired_acc))
        )

        rospy.loginfo_throttle(
            1.0,
            "v_ref=%.3f, v=%.3f, error=%.3f, desired_acc=%.3f",
            self.v_ref,
            self.v,
            velocity_error,
            desired_acc
        )


def main():
    rospy.init_node(
        "leader_velocity_controller",
        anonymous=False
    )

    controller = LeaderVelocityController()

    control_rate = float(
        rospy.get_param("~control_rate", 10.0)
    )

    rate = rospy.Rate(control_rate)

    while not rospy.is_shutdown():
        controller.update()
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
