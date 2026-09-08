#!/usr/bin/env python3

import math
import random
import time

import rospy

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32


class VehicleViconState:
    """
    Maintains a causal estimate of:
        x, y, yaw
        longitudinal velocity
        longitudinal acceleration

    Velocity is obtained from successive Vicon positions.
    Acceleration is obtained from successive longitudinal velocity estimates.

    Both velocity and acceleration can optionally be low-pass filtered.
    """

    def __init__(self, velocity_cutoff_hz=3.0, acceleration_cutoff_hz=2.0):

        self.velocity_cutoff_hz = velocity_cutoff_hz
        self.acceleration_cutoff_hz = acceleration_cutoff_hz

        self.x = None
        self.y = None
        self.yaw = None

        self.previous_x = None
        self.previous_y = None
        self.previous_time = None

        self.velocity = None
        self.acceleration = None

        self.previous_velocity = None

        self.initialized = False

    @staticmethod
    def quaternion_to_yaw(q):
        """
        Convert quaternion to yaw without requiring tf.
        """

        x = q.x
        y = q.y
        z = q.z
        w = q.w

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def low_pass(new_value, previous_value, cutoff_hz, dt):
        """
        First-order causal low-pass filter.

        cutoff_hz <= 0 disables filtering.
        """

        if previous_value is None:
            return new_value

        if cutoff_hz <= 0.0:
            return new_value

        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        alpha = dt / (rc + dt)

        return previous_value + alpha * (new_value - previous_value)

    def update(self, msg):

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        x = float(position.x)
        y = float(position.y)

        yaw = self.quaternion_to_yaw(orientation)

        # ------------------------------------------------------------
        # Use Vicon message timestamp where available
        # ------------------------------------------------------------

        stamp = msg.header.stamp.to_sec()

        if stamp <= 0.0:
            stamp = rospy.Time.now().to_sec()

        # ------------------------------------------------------------
        # First measurement: only initialize position
        # ------------------------------------------------------------

        if self.previous_time is None:

            self.x = x
            self.y = y
            self.yaw = yaw

            self.previous_x = x
            self.previous_y = y
            self.previous_time = stamp

            return

        dt = stamp - self.previous_time

        # Reject invalid timing
        if dt <= 0.0:
            return

        # Also reject abnormally large gaps because they produce
        # meaningless derivatives.
        if dt > 0.2:
            self.previous_x = x
            self.previous_y = y
            self.previous_time = stamp

            self.x = x
            self.y = y
            self.yaw = yaw

            return

        # ============================================================
        # WORLD-FRAME VELOCITY FROM POSITION DIFFERENCE
        # ============================================================

        vx_world = (x - self.previous_x) / dt
        vy_world = (y - self.previous_y) / dt

        # ============================================================
        # PROJECT VELOCITY ONTO VEHICLE LONGITUDINAL AXIS
        #
        # Important when driving in a circle:
        # global vx alone is NOT the vehicle forward velocity.
        # ============================================================

        velocity_raw = (
            vx_world * math.cos(yaw)
            +
            vy_world * math.sin(yaw)
        )

        velocity_filtered = self.low_pass(
            velocity_raw,
            self.velocity,
            self.velocity_cutoff_hz,
            dt
        )

        # ============================================================
        # LONGITUDINAL ACCELERATION
        # ============================================================

        if self.previous_velocity is None:
            acceleration_raw = 0.0
        else:
            acceleration_raw = (
                velocity_filtered - self.previous_velocity
            ) / dt

        acceleration_filtered = self.low_pass(
            acceleration_raw,
            self.acceleration,
            self.acceleration_cutoff_hz,
            dt
        )

        # ============================================================
        # Store current state
        # ============================================================

        self.x = x
        self.y = y
        self.yaw = yaw

        self.velocity = velocity_filtered
        self.acceleration = acceleration_filtered

        self.previous_velocity = velocity_filtered

        self.previous_x = x
        self.previous_y = y
        self.previous_time = stamp

        self.initialized = True


class ViconOCOMeasurements:

    def __init__(self):

        # ============================================================
        # Vehicle numbers
        # ============================================================

        self.leader_number = int(
            rospy.get_param("~leader_number", 1)
        )

        self.follower_number = int(
            rospy.get_param("~follower_number", 2)
        )

        # ============================================================
        # Raw Vicon input topics
        # ============================================================

        self.leader_vicon_topic = rospy.get_param(
            "~leader_vicon_topic",
            "/vicon/jetracer{}".format(self.leader_number)
        )

        self.follower_vicon_topic = rospy.get_param(
            "~follower_vicon_topic",
            "/vicon/jetracer{}".format(self.follower_number)
        )

        # ============================================================
        # Platoon parameters
        # ============================================================

        self.standstill_distance = float(
            rospy.get_param("~standstill_distance", 1.0)
        )

        self.headway = float(
            rospy.get_param("~headway", 0.5)
        )

        # ============================================================
        # Filtering parameters
        #
        # These are deliberately ROS parameters so we can tune them
        # later without changing the node.
        #
        # Set a cutoff <= 0 to disable that filter.
        # ============================================================

        self.velocity_cutoff_hz = float(
            rospy.get_param("~velocity_cutoff_hz", 3.0)
        )

        self.acceleration_cutoff_hz = float(
            rospy.get_param("~acceleration_cutoff_hz", 2.0)
        )

        # ============================================================
        # OCO/controller publication rate
        # ============================================================

        self.publish_rate = float(
            rospy.get_param("~publish_rate", 10.0)
        )

        # ============================================================
        # Internal Vicon state estimators
        # ============================================================

        self.leader = VehicleViconState(
            self.velocity_cutoff_hz,
            self.acceleration_cutoff_hz
        )

        self.follower = VehicleViconState(
            self.velocity_cutoff_hz,
            self.acceleration_cutoff_hz
        )

        # ============================================================
        # Diagnostic / directly usable physical measurements
        #
        # No namespaces:
        #
        # /vicon_velocity_1
        # /vicon_acceleration_1
        # /vicon_velocity_2
        # /vicon_acceleration_2
        # ============================================================

        self.leader_velocity_pub = rospy.Publisher(
            "/vicon_velocity_{}".format(self.leader_number),
            Float32,
            queue_size=10
        )

        self.leader_acceleration_pub = rospy.Publisher(
            "/vicon_acceleration_{}".format(self.leader_number),
            Float32,
            queue_size=10
        )

        self.follower_velocity_pub = rospy.Publisher(
            "/vicon_velocity_{}".format(self.follower_number),
            Float32,
            queue_size=10
        )

        self.follower_acceleration_pub = rospy.Publisher(
            "/vicon_acceleration_{}".format(self.follower_number),
            Float32,
            queue_size=10
        )

        self.spacing_error_pub = rospy.Publisher(
            "/spacing_error_meas_{}".format(self.follower_number),
            Float32,
            queue_size=10
        )

        self.relative_velocity_pub = rospy.Publisher(
            "/relative_velocity_meas_{}".format(self.follower_number),
            Float32,
            queue_size=10
        )

        self.distance_pub = rospy.Publisher(
            "/distance_meas_{}".format(self.follower_number),
            Float32,
            queue_size=10
        )

        # ============================================================
        # OCO measurement topics
        #
        # Real-hardware convention:
        #
        # /oco/y1_2
        # /oco/y2_2
        # ...
        # /oco/y9_2
        #
        # Later test2/test3 will point the observer/fusion node here.
        # ============================================================

        self.publishers = {}

        for channel in range(1, 10):

            topic = "/oco/y{}_{}".format(
                channel,
                self.follower_number
            )

            self.publishers[channel] = rospy.Publisher(
                topic,
                Float32,
                queue_size=10
            )

        # ============================================================
        # Attack configuration
        #
        # Kept compatible with existing oco_measurement_generator.
        # ============================================================

        self.attack_enabled = rospy.get_param(
            "~attack_enabled",
            False
        )

        self.attack_type = rospy.get_param(
            "~attack_type",
            "none"
        )

        self.attack_channels = rospy.get_param(
            "~attack_channels",
            []
        )

        # Make sure all channel numbers are integers
        self.attack_channels = [
            int(channel)
            for channel in self.attack_channels
        ]

        self.step_amplitude = float(
            rospy.get_param("~step_amplitude", 0.0)
        )

        self.white_noise_std = float(
            rospy.get_param("~white_noise_std", 0.0)
        )

        self.attack_start = float(
            rospy.get_param("~attack_start", 0.0)
        )

        self.attack_stop = float(
            rospy.get_param("~attack_stop", -1.0)
        )

        self.switching_period = float(
            rospy.get_param("~switching_period", 4.0)
        )

        self.switching_on_time = float(
            rospy.get_param("~switching_on_time", 2.0)
        )

        # Use monotonic wall time for attack timing
        self.start_time = time.monotonic()

        # ============================================================
        # Subscribers
        # ============================================================

        rospy.Subscriber(
            self.leader_vicon_topic,
            PoseWithCovarianceStamped,
            self.leader_vicon_callback,
            queue_size=20
        )

        rospy.Subscriber(
            self.follower_vicon_topic,
            PoseWithCovarianceStamped,
            self.follower_vicon_callback,
            queue_size=20
        )

        # ============================================================
        # Publish at controller/OCO rate
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate),
            self.timer_callback
        )

        # ============================================================
        # Information
        # ============================================================

        rospy.loginfo("Vicon OCO measurement bridge started")

        rospy.loginfo(
            "Leader Vicon: %s",
            self.leader_vicon_topic
        )

        rospy.loginfo(
            "Follower Vicon: %s",
            self.follower_vicon_topic
        )

        rospy.loginfo(
            "Leader vehicle number: %d",
            self.leader_number
        )

        rospy.loginfo(
            "Follower vehicle number: %d",
            self.follower_number
        )

        rospy.loginfo(
            "Publish rate: %.2f Hz",
            self.publish_rate
        )

        rospy.loginfo(
            "Velocity LPF cutoff: %.2f Hz",
            self.velocity_cutoff_hz
        )

        rospy.loginfo(
            "Acceleration LPF cutoff: %.2f Hz",
            self.acceleration_cutoff_hz
        )

        rospy.loginfo(
            "Attack enabled: %s",
            str(self.attack_enabled)
        )

        rospy.loginfo(
            "Attack channels: %s",
            str(self.attack_channels)
        )

    # ================================================================
    # Vicon callbacks
    # ================================================================

    def leader_vicon_callback(self, msg):
        self.leader.update(msg)

    def follower_vicon_callback(self, msg):
        self.follower.update(msg)

    # ================================================================
    # Attack functions
    # ================================================================

    def attack_time_active(self, elapsed):

        if elapsed < self.attack_start:
            return False

        if self.attack_stop >= 0.0:
            if elapsed > self.attack_stop:
                return False

        return True

    def switching_active(self, elapsed):

        if self.switching_period <= 0.0:
            return True

        attack_elapsed = elapsed - self.attack_start

        phase = attack_elapsed % self.switching_period

        return phase < self.switching_on_time

    def apply_attack(self, channel, clean_value, elapsed):

        if not self.attack_enabled:
            return clean_value

        if channel not in self.attack_channels:
            return clean_value

        if not self.attack_time_active(elapsed):
            return clean_value

        attack_type = self.attack_type.lower()

        if attack_type == "none":

            return clean_value

        elif attack_type == "white_noise":

            return clean_value + random.gauss(
                0.0,
                self.white_noise_std
            )

        elif attack_type == "switching_white_noise":

            if self.switching_active(elapsed):

                return clean_value + random.gauss(
                    0.0,
                    self.white_noise_std
                )

            return clean_value

        elif attack_type == "step":

            return clean_value + self.step_amplitude

        elif attack_type == "switching_step":

            if self.switching_active(elapsed):
                return clean_value + self.step_amplitude

            return clean_value

        else:

            rospy.logwarn_throttle(
                5.0,
                "Unknown attack type '%s'; publishing clean data.",
                self.attack_type
            )

            return clean_value

    # ================================================================
    # Helper publishing function
    # ================================================================

    @staticmethod
    def publish_float(publisher, value):

        msg = Float32()
        msg.data = float(value)

        publisher.publish(msg)

    # ================================================================
    # Main 10 Hz output
    # ================================================================

    def timer_callback(self, event):

        # Wait until both vehicles have valid velocity/acceleration
        if not self.leader.initialized:
            return

        if not self.follower.initialized:
            return

        # ============================================================
        # Extract physical quantities
        # ============================================================

        v1 = self.leader.velocity
        a1 = self.leader.acceleration

        v2 = self.follower.velocity
        a2 = self.follower.acceleration

        # ------------------------------------------------------------
        # Relative velocity
        #
        # Paper convention:
        # Delta v = v_leader - v_follower
        # ------------------------------------------------------------

        delta_v = v1 - v2

        # ------------------------------------------------------------
        # Euclidean inter-vehicle distance from Vicon
        #
        # This matches the distance approach currently used in the
        # simulator measurement setup.
        # ------------------------------------------------------------

        dx = self.leader.x - self.follower.x
        dy = self.leader.y - self.follower.y

        distance = math.sqrt(
            dx * dx + dy * dy
        )

        # ------------------------------------------------------------
        # Constant-time-headway spacing error
        #
        # e = d - s - h*v2
        # ------------------------------------------------------------

        spacing_error = (
            distance
            - self.standstill_distance
            - self.headway * v2
        )

        # ============================================================
        # Publish clean physical measurements
        # ============================================================

        self.publish_float(
            self.leader_velocity_pub,
            v1
        )

        self.publish_float(
            self.leader_acceleration_pub,
            a1
        )

        self.publish_float(
            self.follower_velocity_pub,
            v2
        )

        self.publish_float(
            self.follower_acceleration_pub,
            a2
        )

        self.publish_float(
            self.spacing_error_pub,
            spacing_error
        )

        self.publish_float(
            self.relative_velocity_pub,
            delta_v
        )

        self.publish_float(
            self.distance_pub,
            distance
        )

        # ============================================================
        # Construct the full nine-channel OCO measurement set
        #
        # Current implementation:
        #
        # y1 = spacing error
        # y2 = follower velocity
        # y3 = follower acceleration
        # y4 = leader velocity - follower velocity
        # y5 = leader acceleration
        # y6 = spacing error
        # y7 = follower velocity
        # y8 = spacing error
        # y9 = follower velocity
        # ============================================================

        clean_measurements = {

            1: spacing_error,
            2: v2,
            3: a2,
            4: delta_v,
            5: a1,
            6: spacing_error,
            7: v2,
            8: spacing_error,
            9: v2,
        }

        elapsed = time.monotonic() - self.start_time

        # ============================================================
        # Apply optional attacks and publish y1 ... y9
        # ============================================================

        for channel, clean_value in clean_measurements.items():

            output_value = self.apply_attack(
                channel,
                clean_value,
                elapsed
            )

            self.publish_float(
                self.publishers[channel],
                output_value
            )


if __name__ == "__main__":

    rospy.init_node("vicon_oco_measurements")

    node = ViconOCOMeasurements()

    rospy.spin()
