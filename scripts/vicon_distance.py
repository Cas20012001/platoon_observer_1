#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32


class ViconDistance:
    def __init__(self):
        self.leader_x = None
        self.leader_y = None
        self.follower_x = None
        self.follower_y = None
        self.follower_velocity = None

        self.s = rospy.get_param("~s", 1.0)
        self.h = rospy.get_param("~h", 0.5)

        leader_topic = rospy.get_param(
            "~leader_vicon_topic",
            "/vicon/jetracer1"
        )

        follower_topic = rospy.get_param(
            "~follower_vicon_topic",
            "/vicon/jetracer2"
        )

        velocity_topic = rospy.get_param(
            "~follower_velocity_topic",
            "/vicon_velocity_meas_2"
        )

        distance_topic = rospy.get_param(
            "~distance_topic",
            "/distance_meas_2"
        )

        error_topic = rospy.get_param(
            "~error_topic",
            "/spacing_error_meas_2"
        )

        self.distance_pub = rospy.Publisher(
            distance_topic,
            Float32,
            queue_size=1
        )

        self.error_pub = rospy.Publisher(
            error_topic,
            Float32,
            queue_size=1
        )

        rospy.Subscriber(
            leader_topic,
            PoseWithCovarianceStamped,
            self.leader_callback,
            queue_size=1
        )

        rospy.Subscriber(
            follower_topic,
            PoseWithCovarianceStamped,
            self.follower_callback,
            queue_size=1
        )

        rospy.Subscriber(
            velocity_topic,
            Float32,
            self.velocity_callback,
            queue_size=1
        )

    def leader_callback(self, msg):
        self.leader_x = msg.pose.pose.position.x
        self.leader_y = msg.pose.pose.position.y
        self.publish_values()

    def follower_callback(self, msg):
        self.follower_x = msg.pose.pose.position.x
        self.follower_y = msg.pose.pose.position.y
        self.publish_values()

    def velocity_callback(self, msg):
        self.follower_velocity = msg.data
        self.publish_values()

    def publish_values(self):
        if self.leader_x is None or self.follower_x is None:
            return

        dx = self.leader_x - self.follower_x
        dy = self.leader_y - self.follower_y

        distance = math.sqrt(dx * dx + dy * dy)

        self.distance_pub.publish(
            Float32(data=distance)
        )

        if self.follower_velocity is not None:
            spacing_error = (
                distance
                - self.s
                - self.h * self.follower_velocity
            )

            self.error_pub.publish(
                Float32(data=spacing_error)
            )


if __name__ == "__main__":
    rospy.init_node("vicon_distance")
    ViconDistance()
    rospy.spin()
