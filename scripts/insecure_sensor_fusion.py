#!/usr/bin/env python3

import rospy

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray


class InsecureSensorFusion:
    def __init__(self):

        # ============================================================
        # Input topics
        # ============================================================

        self.topic_y1 = rospy.get_param(
            "~y1_topic",
            "/car2/oco/y1"
        )

        self.topic_y2 = rospy.get_param(
            "~y2_topic",
            "/car2/oco/y2"
        )

        self.topic_y3 = rospy.get_param(
            "~y3_topic",
            "/car2/oco/y3"
        )

        self.topic_y4 = rospy.get_param(
            "~y4_topic",
            "/car2/oco/y4"
        )

        self.topic_y5 = rospy.get_param(
            "~y5_topic",
            "/car2/oco/y5"
        )

        self.topic_y6 = rospy.get_param(
            "~y6_topic",
            "/car2/oco/y6"
        )

        self.topic_y7 = rospy.get_param(
            "~y7_topic",
            "/car2/oco/y7"
        )

        self.topic_y8 = rospy.get_param(
            "~y8_topic",
            "/car2/oco/y8"
        )

        self.topic_y9 = rospy.get_param(
            "~y9_topic",
            "/car2/oco/y9"
        )

        # ============================================================
        # Output topic
        #
        # Same state ordering used by the OCO and CACC:
        #
        # [e, v2, a2, delta_v, a1]
        # ============================================================

        self.state_topic = rospy.get_param(
            "~state_topic",
            "/car2/insecure_state_1"
        )

        # ============================================================
        # Publishing rate
        # ============================================================

        self.publish_rate = float(
            rospy.get_param("~publish_rate", 10.0)
        )

        # ============================================================
        # Internal measurement storage
        # ============================================================

        self.y = {
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: None
        }

        # ============================================================
        # Publisher
        # ============================================================

        self.state_pub = rospy.Publisher(
            self.state_topic,
            Float32MultiArray,
            queue_size=10
        )

        # ============================================================
        # Subscribers
        # ============================================================

        rospy.Subscriber(
            self.topic_y1,
            Float32,
            self.y1_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.topic_y2,
            Float32,
            self.y2_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.topic_y3,
            Float32,
            self.y3_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.topic_y4,
            Float32,
            self.y4_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.topic_y5,
            Float32,
            self.y5_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.topic_y6,
            Float32,
            self.y6_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.topic_y7,
            Float32,
            self.y7_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.topic_y8,
            Float32,
            self.y8_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.topic_y9,
            Float32,
            self.y9_callback,
            queue_size=10
        )

        # ============================================================
        # Fixed-rate state publishing
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate),
            self.timer_callback
        )

        rospy.loginfo("Insecure sensor fusion started")
        rospy.loginfo("Output state topic: %s", self.state_topic)

    # ================================================================
    # Measurement callbacks
    # ================================================================

    def y1_callback(self, msg):
        self.y[1] = float(msg.data)

    def y2_callback(self, msg):
        self.y[2] = float(msg.data)

    def y3_callback(self, msg):
        self.y[3] = float(msg.data)

    def y4_callback(self, msg):
        self.y[4] = float(msg.data)

    def y5_callback(self, msg):
        self.y[5] = float(msg.data)

    def y6_callback(self, msg):
        self.y[6] = float(msg.data)

    def y7_callback(self, msg):
        self.y[7] = float(msg.data)

    def y8_callback(self, msg):
        self.y[8] = float(msg.data)

    def y9_callback(self, msg):
        self.y[9] = float(msg.data)

    # ================================================================
    # Main update
    # ================================================================

    def timer_callback(self, event):

        # Wait until all nine measurements have been received.
        for channel in range(1, 10):
            if self.y[channel] is None:
                return

        # ============================================================
        # Paper-style insecure measurement fusion
        #
        # Redundant spacing measurements:
        #
        #   e = (y1 + y6 + y8) / 3
        #
        # Redundant follower velocity measurements:
        #
        #   v2 = (y2 + y7 + y9) / 3
        #
        # Remaining quantities are used directly:
        #
        #   a2      = y3
        #   delta_v = y4
        #   a1      = y5
        #
        # NO attack detection
        # NO observer
        # NO eta
        # NO beta
        # NO sensor rejection
        # ============================================================

        e = (
            self.y[1]
            + self.y[6]
            + self.y[8]
        ) / 3.0

        v2 = (
            self.y[2]
            + self.y[7]
            + self.y[9]
        ) / 3.0

        a2 = self.y[3]

        delta_v = self.y[4]

        a1 = self.y[5]

        # ============================================================
        # Publish in exactly the same state ordering as the OCO
        #
        # x = [e, v2, a2, delta_v, a1]
        # ============================================================

        msg = Float32MultiArray()

        msg.data = [
            e,
            v2,
            a2,
            delta_v,
            a1
        ]

        self.state_pub.publish(msg)


if __name__ == "__main__":

    rospy.init_node("insecure_sensor_fusion")

    node = InsecureSensorFusion()

    rospy.spin()
