#!/usr/bin/env python3

import os
import time

import rospy
from std_msgs.msg import Float32


class LeaderVelocityControllerReal:

    def __init__(self):

        # ============================================================
        # Vehicle
        # ============================================================

        car_number = rospy.get_param(
            "~car_number",
            os.environ.get("car_number", "2")
        )

        self.car_number = str(car_number)

        # ============================================================
        # Controller parameters
        #
        # desired_acc = kp * (v_ref - v)
        # ============================================================

        self.kp = float(
            rospy.get_param("~kp", 1.0)
        )

        self.min_acc = float(
            rospy.get_param("~min_acc", -1.0)
        )

        self.max_acc = float(
            rospy.get_param("~max_acc", 1.0)
        )

        # ============================================================
        # Vicon timeout
        #
        # If no new velocity measurement has arrived within this
        # period, desired acceleration is set to zero.
        #
        # Set <= 0 to disable timeout checking.
        # ============================================================

        self.velocity_timeout = float(
            rospy.get_param("~velocity_timeout", 0.25)
        )

        # ============================================================
        # Internal state
        # ============================================================

        self.v_ref = 0.0
        self.v = 0.0

        self.velocity_received = False
        self.last_velocity_time = None

        # ============================================================
        # Topics
        # ============================================================

        velocity_topic = rospy.get_param(
            "~velocity_topic",
            "/vicon_velocity_" + self.car_number
        )

        v_ref_topic = rospy.get_param(
            "~v_ref_topic",
            "/v_ref_" + self.car_number
        )

        desired_acc_topic = rospy.get_param(
            "~desired_acc_topic",
            "/desired_acc_" + self.car_number
        )

        # ============================================================
        # Publisher
        # ============================================================

        self.desired_acc_pub = rospy.Publisher(
            desired_acc_topic,
            Float32,
            queue_size=1
        )

        # ============================================================
        # Subscribers
        # ============================================================

        rospy.Subscriber(
            velocity_topic,
            Float32,
            self.velocity_callback,
            queue_size=1
        )

        rospy.Subscriber(
            v_ref_topic,
            Float32,
            self.v_ref_callback,
            queue_size=1
        )

        # ============================================================
        # Startup information
        # ============================================================

        rospy.loginfo(
            "Real leader velocity controller started for car %s",
            self.car_number
        )

        rospy.loginfo(
            "Vicon velocity topic: %s",
            velocity_topic
        )

        rospy.loginfo(
            "Velocity reference topic: %s",
            v_ref_topic
        )

        rospy.loginfo(
            "Desired acceleration topic: %s",
            desired_acc_topic
        )

        rospy.loginfo(
            "kp=%.3f, acceleration limits=[%.3f, %.3f]",
            self.kp,
            self.min_acc,
            self.max_acc
        )

        rospy.loginfo(
            "Velocity timeout: %.3f s",
            self.velocity_timeout
        )

    # ================================================================
    # Callbacks
    # ================================================================

    def velocity_callback(self, msg):

        self.v = float(msg.data)

        self.velocity_received = True

        # Monotonic time is deliberately used here.
        # It is unaffected by system/ROS clock corrections.
        self.last_velocity_time = time.monotonic()

    def v_ref_callback(self, msg):

        self.v_ref = float(msg.data)

    # ================================================================
    # Main controller
    # ================================================================

    def update(self):

        # ------------------------------------------------------------
        # Wait for first Vicon velocity
        # ------------------------------------------------------------

        if not self.velocity_received:

            rospy.logwarn_throttle(
                2.0,
                "Waiting for Vicon velocity measurement"
            )

            return

        # ------------------------------------------------------------
        # Check for stale Vicon data
        # ------------------------------------------------------------

        if self.velocity_timeout > 0.0:

            age = (
                time.monotonic()
                - self.last_velocity_time
            )

            if age > self.velocity_timeout:

                rospy.logwarn_throttle(
                    1.0,
                    "Vicon velocity stale (age %.3f s). "
                    "Publishing zero desired acceleration.",
                    age
                )

                self.desired_acc_pub.publish(
                    Float32(data=0.0)
                )

                return

        # ============================================================
        # Proportional velocity controller
        # ============================================================

        velocity_error = self.v_ref - self.v

        desired_acc = (
            self.kp * velocity_error
        )

        # ============================================================
        # Acceleration saturation
        # ============================================================

        desired_acc = max(
            self.min_acc,
            min(
                self.max_acc,
                desired_acc
            )
        )

        # ============================================================
        # Publish desired acceleration
        # ============================================================

        self.desired_acc_pub.publish(
            Float32(
                data=float(desired_acc)
            )
        )

        # ============================================================
        # Diagnostic output
        # ============================================================

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
        "leader_velocity_controller_real",
        anonymous=False
    )

    controller = LeaderVelocityControllerReal()

    control_rate = float(
        rospy.get_param(
            "~control_rate",
            10.0
        )
    )

    rate = rospy.Rate(
        control_rate
    )

    while not rospy.is_shutdown():

        controller.update()

        rate.sleep()


if __name__ == "__main__":

    try:

        main()

    except rospy.ROSInterruptException:

        pass
