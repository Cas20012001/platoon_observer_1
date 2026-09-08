#!/usr/bin/env python3

import rospy
import random
import math

from std_msgs.msg import Float32


class OCOMeasurementGenerator:
    def __init__(self):

        # ============================================================
        # Input topics
        # ============================================================

        self.spacing_topic = rospy.get_param(
            "~spacing_topic",
            "/car2/spacing_error_meas_1"
        )

        # Follower longitudinal velocity
        self.velocity_topic = rospy.get_param(
            "~velocity_topic",
            "/car2/vx_1"
        )

        # Leader longitudinal velocity
        self.leader_velocity_topic = rospy.get_param(
            "~leader_velocity_topic",
            "/car1/vx_1"
        )

        # ============================================================
        # Output topics
        #
        # Current implementation:
        #
        # y1, y6, y8 = spacing-error related
        # y2, y7, y9 = follower velocity related
        # y3         = follower acceleration
        # y4         = relative velocity v1 - v2
        # y5         = leader acceleration
        #
        # Note:
        # y3 and y5 are derived from the measured velocity signals
        # using the ACTUAL elapsed time between velocity callbacks.
        # ============================================================

        self.publishers = {
            1: rospy.Publisher("/car2/oco/y1", Float32, queue_size=10),
            2: rospy.Publisher("/car2/oco/y2", Float32, queue_size=10),
            3: rospy.Publisher("/car2/oco/y3", Float32, queue_size=10),
            4: rospy.Publisher("/car2/oco/y4", Float32, queue_size=10),
            5: rospy.Publisher("/car2/oco/y5", Float32, queue_size=10),
            6: rospy.Publisher("/car2/oco/y6", Float32, queue_size=10),
            7: rospy.Publisher("/car2/oco/y7", Float32, queue_size=10),
            8: rospy.Publisher("/car2/oco/y8", Float32, queue_size=10),
            9: rospy.Publisher("/car2/oco/y9", Float32, queue_size=10),
        }

        # ============================================================
        # Sampling
        # ============================================================

        self.publish_rate = rospy.get_param("~publish_rate", 10.0)

        # Minimum allowed dt for numerical differentiation.
        # This only protects against duplicate/nearly simultaneous
        # callbacks causing division by a very small number.
        self.min_derivative_dt = float(
            rospy.get_param("~min_derivative_dt", 1e-4)
        )

        # ============================================================
        # Attack configuration
        #
        # attack_type:
        #
        # none
        # white_noise
        # switching_white_noise
        # step
        # switching_step
        #
        # These correspond to the attack categories used in the paper.
        # ============================================================

        self.attack_enabled = rospy.get_param(
            "~attack_enabled",
            False
        )

        self.attack_type = rospy.get_param(
            "~attack_type",
            "none"
        )

        # Example:
        # attack_channels: [1, 2]
        #
        # means y1 and y2 are attacked.
        self.attack_channels = rospy.get_param(
            "~attack_channels",
            []
        )

        # Step attack magnitude
        self.step_amplitude = rospy.get_param(
            "~step_amplitude",
            0.0
        )

        # Standard deviation of additive white noise
        self.white_noise_std = rospy.get_param(
            "~white_noise_std",
            0.0
        )

        # Attack starts this many seconds after node starts
        self.attack_start = rospy.get_param(
            "~attack_start",
            0.0
        )

        # Negative value means attack never automatically stops
        self.attack_stop = rospy.get_param(
            "~attack_stop",
            -1.0
        )

        # Used for repeatedly switching attacks
        #
        # Example:
        # switching_period = 4 s
        # switching_on_time = 2 s
        #
        # ON  for first 2 s
        # OFF for next 2 s
        # repeat
        self.switching_period = rospy.get_param(
            "~switching_period",
            4.0
        )

        self.switching_on_time = rospy.get_param(
            "~switching_on_time",
            2.0
        )

        # ============================================================
        # Internal measurement storage
        # ============================================================

        self.spacing_error = None

        self.velocity = None
        self.leader_velocity = None

        # Latest derived accelerations
        self.follower_acceleration = None
        self.leader_acceleration = None

        # Previous velocity samples used for differentiation
        self.prev_follower_velocity = None
        self.prev_leader_velocity = None

        # Time at which previous velocity samples arrived
        self.prev_follower_velocity_time = None
        self.prev_leader_velocity_time = None

        self.start_time = rospy.Time.now()

        # ============================================================
        # Subscribers
        # ============================================================

        rospy.Subscriber(
            self.spacing_topic,
            Float32,
            self.spacing_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.velocity_topic,
            Float32,
            self.velocity_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.leader_velocity_topic,
            Float32,
            self.leader_velocity_callback,
            queue_size=10
        )

        # ============================================================
        # Fixed-rate publishing
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate),
            self.timer_callback
        )

        rospy.loginfo("OCO measurement generator started")

        rospy.loginfo(
            "Spacing source: %s",
            self.spacing_topic
        )

        rospy.loginfo(
            "Follower velocity source: %s",
            self.velocity_topic
        )

        rospy.loginfo(
            "Leader velocity source: %s",
            self.leader_velocity_topic
        )

        rospy.loginfo(
            "Attack enabled: %s",
            self.attack_enabled
        )

        rospy.loginfo(
            "Attack type: %s",
            self.attack_type
        )

        rospy.loginfo(
            "Attacked channels: %s",
            str(self.attack_channels)
        )

    # ================================================================
    # Input callbacks
    # ================================================================

    def spacing_callback(self, msg):
        self.spacing_error = float(msg.data)

    def velocity_callback(self, msg):
        """
        Follower velocity callback.

        Besides storing v2, derive the actual follower acceleration:

            a2 = dv2 / dt

        where dt is measured from the actual callback arrival times.
        """

        current_velocity = float(msg.data)
        current_time = rospy.Time.now()

        if (
            self.prev_follower_velocity is not None
            and self.prev_follower_velocity_time is not None
        ):
            dt = (
                current_time - self.prev_follower_velocity_time
            ).to_sec()

            if dt >= self.min_derivative_dt:

                self.follower_acceleration = (
                    current_velocity
                    - self.prev_follower_velocity
                ) / dt

        self.prev_follower_velocity = current_velocity
        self.prev_follower_velocity_time = current_time

        self.velocity = current_velocity

    def leader_velocity_callback(self, msg):
        """
        Leader velocity callback.

        Besides storing v1, derive the actual leader acceleration:

            a1 = dv1 / dt

        where dt is measured from the actual callback arrival times.
        """

        current_velocity = float(msg.data)
        current_time = rospy.Time.now()

        if (
            self.prev_leader_velocity is not None
            and self.prev_leader_velocity_time is not None
        ):
            dt = (
                current_time - self.prev_leader_velocity_time
            ).to_sec()

            if dt >= self.min_derivative_dt:

                self.leader_acceleration = (
                    current_velocity
                    - self.prev_leader_velocity
                ) / dt

        self.prev_leader_velocity = current_velocity
        self.prev_leader_velocity_time = current_time

        self.leader_velocity = current_velocity

    # ================================================================
    # Attack logic
    # ================================================================

    def attack_time_active(self, elapsed):
        """
        Check whether we are inside the overall attack interval.
        """

        if elapsed < self.attack_start:
            return False

        if self.attack_stop >= 0.0:
            if elapsed > self.attack_stop:
                return False

        return True

    def switching_active(self, elapsed):
        """
        Determine whether a switching attack is currently in its ON phase.
        """

        if self.switching_period <= 0.0:
            return True

        attack_elapsed = elapsed - self.attack_start

        phase = attack_elapsed % self.switching_period

        return phase < self.switching_on_time

    def apply_attack(self, channel, clean_value, elapsed):

        # Attack system globally disabled
        if not self.attack_enabled:
            return clean_value

        # Channel is not attacked
        if channel not in self.attack_channels:
            return clean_value

        # Outside attack time interval
        if not self.attack_time_active(elapsed):
            return clean_value

        attack_type = self.attack_type.lower()

        # ------------------------------------------------------------
        # No attack
        # ------------------------------------------------------------

        if attack_type == "none":
            return clean_value

        # ------------------------------------------------------------
        # White-noise attack
        # ------------------------------------------------------------

        elif attack_type == "white_noise":

            attack = random.gauss(
                0.0,
                self.white_noise_std
            )

            return clean_value + attack

        # ------------------------------------------------------------
        # Repeatedly switching white-noise attack
        # ------------------------------------------------------------

        elif attack_type == "switching_white_noise":

            if self.switching_active(elapsed):

                attack = random.gauss(
                    0.0,
                    self.white_noise_std
                )

                return clean_value + attack

            return clean_value

        # ------------------------------------------------------------
        # Step attack
        # ------------------------------------------------------------

        elif attack_type == "step":

            return clean_value + self.step_amplitude

        # ------------------------------------------------------------
        # Repeatedly switching step attack
        # ------------------------------------------------------------

        elif attack_type == "switching_step":

            if self.switching_active(elapsed):
                return clean_value + self.step_amplitude

            return clean_value

        # ------------------------------------------------------------
        # Invalid attack type
        # ------------------------------------------------------------

        else:

            rospy.logwarn_throttle(
                5.0,
                "Unknown attack type '%s'. Publishing clean data.",
                self.attack_type
            )

            return clean_value

    # ================================================================
    # Main update
    # ================================================================

    def timer_callback(self, event):

        # ------------------------------------------------------------
        # Wait until all required source measurements are available.
        #
        # The acceleration signals need at least TWO velocity samples
        # before their first valid derivative can be calculated.
        # ------------------------------------------------------------

        if self.spacing_error is None:
            return

        if self.velocity is None:
            return

        if self.leader_velocity is None:
            return

        if self.follower_acceleration is None:
            return

        if self.leader_acceleration is None:
            return

        elapsed = (
            rospy.Time.now() - self.start_time
        ).to_sec()

        # ============================================================
        # Construct clean measurements
        #
        # y1 = spacing error
        #
        # y2 = follower velocity v2
        #
        # y3 = follower actual acceleration a2
        #
        # y4 = relative velocity
        #      delta_v = v1 - v2
        #
        # y5 = leader actual acceleration a1
        #
        # y6 = spacing error
        # y7 = follower velocity
        # y8 = spacing error
        # y9 = follower velocity
        # ============================================================

        relative_velocity = (
            self.leader_velocity - self.velocity
        )

        clean_measurements = {
            1: self.spacing_error,
            2: self.velocity,
            3: self.follower_acceleration,
            4: relative_velocity,
            5: self.leader_acceleration,
            6: self.spacing_error,
            7: self.velocity,
            8: self.spacing_error,
            9: self.velocity,
        }

        # ============================================================
        # Apply optional attacks and publish
        # ============================================================

        for channel, clean_value in clean_measurements.items():

            attacked_value = self.apply_attack(
                channel,
                clean_value,
                elapsed
            )

            msg = Float32()
            msg.data = attacked_value

            self.publishers[channel].publish(msg)


if __name__ == "__main__":

    rospy.init_node("oco_measurement_generator")

    node = OCOMeasurementGenerator()

    rospy.spin()
