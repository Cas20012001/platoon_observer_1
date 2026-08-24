#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32


class ViconVelocity:
    def __init__(self):
        self.vicon_topic = rospy.get_param(
            "~vicon_topic",
            "/vicon/jetracer2"
        )

        self.velocity_topic = rospy.get_param(
            "~velocity_topic",
            "/vicon_velocity_meas_2"
        )

        self.velocity_alpha = rospy.get_param(
            "~velocity_alpha",
            0.2
        )

        self.last_x = None
        self.last_y = None
        self.last_stamp = None

        self.vicon_velocity = 0.0
        self.have_velocity = False

        self.velocity_pub = rospy.Publisher(
            self.velocity_topic,
            Float32,
            queue_size=1
        )

        rospy.Subscriber(
            self.vicon_topic,
            PoseWithCovarianceStamped,
            self.vicon_callback,
            queue_size=1
        )

    def vicon_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        stamp = msg.header.stamp

        if stamp.to_sec() == 0.0:
            stamp = rospy.Time.now()

        if self.last_stamp is not None:
            dt = (stamp - self.last_stamp).to_sec()

            if dt > 0.0001:
                dx = x - self.last_x
                dy = y - self.last_y

                # No-slip assumption: planar speed equals longitudinal speed.
                raw_velocity = math.sqrt(dx * dx + dy * dy) / dt

                if not self.have_velocity:
                    self.vicon_velocity = raw_velocity
                    self.have_velocity = True
                else:
                    alpha = self.velocity_alpha
                    self.vicon_velocity = (
                        alpha * raw_velocity
                        + (1.0 - alpha) * self.vicon_velocity
                    )

                self.velocity_pub.publish(
                    Float32(data=self.vicon_velocity)
                )

        self.last_x = x
        self.last_y = y
        self.last_stamp = stamp


if __name__ == "__main__":
    rospy.init_node("vicon_velocity")

    ViconVelocity()

    rospy.spin()
