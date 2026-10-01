#!/usr/bin/env python3

import rospy

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray


class InsecureStateBuilder:

    def __init__(self):

        # ============================================================
        # Parameters
        # ============================================================

        self.update_rate = float(
            rospy.get_param(
                "~update_rate",
                10.0
            )
        )

        self.state_topic = rospy.get_param(
            "~state_topic",
            "/state_estimate_3"
        )

        # ============================================================
        # Nine sensor channels
        # ============================================================

        self.y_topics = {
            1: rospy.get_param("~y1_topic", "/oco_onboard/y1_3"),
            2: rospy.get_param("~y2_topic", "/oco_onboard/y2_3"),
            3: rospy.get_param("~y3_topic", "/oco_onboard/y3_3"),
            4: rospy.get_param("~y4_topic", "/oco_onboard/y4_3"),
            5: rospy.get_param("~y5_topic", "/oco_onboard/y5_3"),
            6: rospy.get_param("~y6_topic", "/oco_onboard/y6_3"),
            7: rospy.get_param("~y7_topic", "/oco_onboard/y7_3"),
            8: rospy.get_param("~y8_topic", "/oco_onboard/y8_3"),
            9: rospy.get_param("~y9_topic", "/oco_onboard/y9_3"),
        }

        # ============================================================
        # Latest measurements
        # ============================================================

        self.y = {
            channel: None
            for channel in range(1, 10)
        }

        # ============================================================
        # Subscribers
        # ============================================================

        for channel in range(1, 10):

            rospy.Subscriber(
                self.y_topics[channel],
                Float32,
                self.measurement_callback,
                callback_args=channel,
                queue_size=10
            )

        # ============================================================
        # State publisher
        #
        # Same format as secure OCO:
        #
        # [
        #   e,
        #   v2,
        #   a2,
        #   delta_v,
        #   a1
        # ]
        # ============================================================

        self.state_publisher = rospy.Publisher(
            self.state_topic,
            Float32MultiArray,
            queue_size=10
        )

        # ============================================================
        # Update timer
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0 / self.update_rate
            ),
            self.update_callback
        )

        rospy.loginfo(
            "========================================"
        )

        rospy.loginfo(
            "INSECURE direct-measurement state builder started"
        )

        rospy.loginfo(
            "State output: %s",
            self.state_topic
        )

        rospy.loginfo(
            "========================================"
        )


    # ================================================================
    # Measurement callback
    # ================================================================

    def measurement_callback(
        self,
        msg,
        channel
    ):

        self.y[channel] = float(
            msg.data
        )


    # ================================================================
    # Check whether all sensors have been received
    # ================================================================

    def data_ready(self):

        for channel in range(1, 10):

            if self.y[channel] is None:
                return False

        return True


    # ================================================================
    # Build insecure state
    # ================================================================

    def update_callback(self, event):

        if not self.data_ready():
            return

        # ------------------------------------------------------------
        # Spacing error
        #
        # Three redundant measurements:
        #
        # y1, y6, y8
        #
        # Insecure baseline simply averages them.
        # ------------------------------------------------------------

        e = (
            self.y[1]
            + self.y[6]
            + self.y[8]
        ) / 3.0

        # ------------------------------------------------------------
        # Follower velocity
        #
        # Three redundant measurements:
        #
        # y2, y7, y9
        #
        # Insecure baseline simply averages them.
        # ------------------------------------------------------------

        v2 = (
            self.y[2]
            + self.y[7]
            + self.y[9]
        ) / 3.0

        # ------------------------------------------------------------
        # Non-redundant measurements
        # ------------------------------------------------------------

        a2 = self.y[3]

        delta_v = self.y[4]

        a1 = self.y[5]

        # ------------------------------------------------------------
        # Construct state
        #
        # x = [
        #   e,
        #   v2,
        #   a2,
        #   delta_v,
        #   a1
        # ]
        # ------------------------------------------------------------

        state_msg = Float32MultiArray()

        state_msg.data = [
            e,
            v2,
            a2,
            delta_v,
            a1
        ]

        self.state_publisher.publish(
            state_msg
        )


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":

    rospy.init_node(
        "insecure_state_builder"
    )

    node = InsecureStateBuilder()

    rospy.spin()
