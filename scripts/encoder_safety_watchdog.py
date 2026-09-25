#!/usr/bin/env python3

import os
import rospy

from std_msgs.msg import Float32, Float32MultiArray


class EncoderSafetyWatchdog:

    def __init__(self):

        self.car_number = str(
            rospy.get_param(
                "~car_number",
                os.environ.get("car_number", "3")
            )
        )

        self.max_speed = float(
            rospy.get_param(
                "~max_speed",
                1.5
            )
        )

        self.encoder_timeout = float(
            rospy.get_param(
                "~encoder_timeout",
                0.25
            )
        )

        self.publish_rate = float(
            rospy.get_param(
                "~publish_rate",
                20.0
            )
        )

        self.encoder_velocity = 0.0
        self.last_encoder_time = None
        self.have_encoder = False

        self.sensor_topic = (
            "/arduino_data_" + self.car_number
        )

        self.safety_topic = (
            "/safety_value_" + self.car_number
        )

        self.safety_pub = rospy.Publisher(
            self.safety_topic,
            Float32,
            queue_size=1
        )

        rospy.Subscriber(
            self.sensor_topic,
            Float32MultiArray,
            self.sensor_callback,
            queue_size=1
        )

        rospy.on_shutdown(
            self.shutdown
        )

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0 / self.publish_rate
            ),
            self.timer_callback
        )

        rospy.loginfo(
            "Encoder safety watchdog car %s started: "
            "max_speed=%.2f m/s, timeout=%.2f s",
            self.car_number,
            self.max_speed,
            self.encoder_timeout
        )


    def sensor_callback(self, msg):

        # arduino_data:
        # [acc_x, acc_y, gyro_z, encoder_velocity]

        if len(msg.data) < 4:

            rospy.logwarn_throttle(
                2.0,
                "Safety watchdog car %s: "
                "arduino_data message too short",
                self.car_number
            )

            return

        self.encoder_velocity = float(
            msg.data[3]
        )

        self.last_encoder_time = (
            rospy.Time.now()
        )

        self.have_encoder = True


    def timer_callback(self, event):

        safety = 0.0

        # --------------------------------------------------------
        # Condition 1:
        # encoder must have been received at least once
        # --------------------------------------------------------

        if not self.have_encoder:

            rospy.logwarn_throttle(
                2.0,
                "Safety car %s: waiting for encoder",
                self.car_number
            )

            self.safety_pub.publish(
                Float32(data=0.0)
            )

            return


        # --------------------------------------------------------
        # Condition 2:
        # encoder data must be recent
        # --------------------------------------------------------

        age = (
            rospy.Time.now()
            - self.last_encoder_time
        ).to_sec()

        if age > self.encoder_timeout:

            rospy.logwarn_throttle(
                1.0,
                "SAFETY STOP car %s: "
                "encoder stale (%.3f s)",
                self.car_number,
                age
            )

            self.safety_pub.publish(
                Float32(data=0.0)
            )

            return


        # --------------------------------------------------------
        # Condition 3:
        # speed must remain below hard limit
        # --------------------------------------------------------

        if abs(self.encoder_velocity) >= self.max_speed:

            rospy.logwarn_throttle(
                1.0,
                "SAFETY STOP car %s: "
                "speed %.3f m/s >= %.3f m/s",
                self.car_number,
                self.encoder_velocity,
                self.max_speed
            )

            self.safety_pub.publish(
                Float32(data=0.0)
            )

            return


        # All conditions satisfied
        safety = 1.0

        self.safety_pub.publish(
            Float32(data=safety)
        )


    def shutdown(self):

        rospy.loginfo(
            "Safety watchdog car %s shutting down: "
            "setting safety to zero",
            self.car_number
        )

        self.safety_pub.publish(
            Float32(data=0.0)
        )


if __name__ == "__main__":

    rospy.init_node(
        "encoder_safety_watchdog"
    )

    EncoderSafetyWatchdog()

    rospy.spin()
