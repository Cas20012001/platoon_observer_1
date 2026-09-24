#!/usr/bin/env python3

import math
import random
import time

import rospy

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32, Float32MultiArray


# ====================================================================
# VICON STATE
# ====================================================================

class VehicleViconState:
    """
    Maintains a causal estimate of:

        x, y, yaw
        longitudinal velocity
        longitudinal acceleration

    Velocity is obtained from successive Vicon positions.

    Acceleration is obtained from successive longitudinal
    velocity estimates.

    Both velocity and acceleration are causally low-pass filtered.
    """

    def __init__(
        self,
        velocity_cutoff_hz=3.0,
        acceleration_cutoff_hz=2.0
    ):

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

        x = q.x
        y = q.y
        z = q.z
        w = q.w

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )


    @staticmethod
    def low_pass(
        new_value,
        previous_value,
        cutoff_hz,
        dt
    ):
        """
        First-order causal low-pass filter.

        cutoff_hz <= 0 disables filtering.
        """

        if previous_value is None:
            return new_value

        if cutoff_hz <= 0.0:
            return new_value

        rc = 1.0 / (
            2.0
            * math.pi
            * cutoff_hz
        )

        alpha = dt / (
            rc + dt
        )

        return (
            previous_value
            + alpha
            * (
                new_value
                - previous_value
            )
        )


    def update(self, msg):

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        x = float(position.x)
        y = float(position.y)

        yaw = self.quaternion_to_yaw(
            orientation
        )

        # ------------------------------------------------------------
        # Use Vicon timestamp where available
        # ------------------------------------------------------------

        stamp = msg.header.stamp.to_sec()

        if stamp <= 0.0:
            stamp = rospy.Time.now().to_sec()

        # ------------------------------------------------------------
        # First measurement
        # ------------------------------------------------------------

        if self.previous_time is None:

            self.x = x
            self.y = y
            self.yaw = yaw

            self.previous_x = x
            self.previous_y = y
            self.previous_time = stamp

            return

        dt = (
            stamp
            - self.previous_time
        )

        if dt <= 0.0:
            return

        # Reject large gaps before differentiating
        if dt > 0.2:

            self.previous_x = x
            self.previous_y = y
            self.previous_time = stamp

            self.x = x
            self.y = y
            self.yaw = yaw

            return

        # ============================================================
        # WORLD-FRAME VELOCITY
        # ============================================================

        vx_world = (
            x
            - self.previous_x
        ) / dt

        vy_world = (
            y
            - self.previous_y
        ) / dt

        # ============================================================
        # PROJECT VELOCITY ON VEHICLE LONGITUDINAL AXIS
        # ============================================================

        velocity_raw = (
            vx_world
            * math.cos(yaw)
            +
            vy_world
            * math.sin(yaw)
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
                velocity_filtered
                - self.previous_velocity
            ) / dt

        acceleration_filtered = self.low_pass(
            acceleration_raw,
            self.acceleration,
            self.acceleration_cutoff_hz,
            dt
        )

        # ============================================================
        # STORE
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


# ====================================================================
# ONBOARD SENSOR STATE
# ====================================================================

class OnboardSensorState:
    """
    Processes /sensors_and_input_N.

    Expected layout:

        data[3] = IMU longitudinal acceleration
        data[6] = encoder velocity

    Both signals receive a causal first-order LPF.

    IMU bias subtraction is included but biases are currently
    configured as zero from the launch file.
    """

    def __init__(
        self,
        velocity_cutoff_hz=2.0,
        acceleration_cutoff_hz=2.0,
        imu_bias=0.0
    ):

        self.velocity_cutoff_hz = velocity_cutoff_hz
        self.acceleration_cutoff_hz = acceleration_cutoff_hz

        self.imu_bias = imu_bias

        self.velocity_raw = None
        self.acceleration_raw = None

        self.velocity = None
        self.acceleration = None

        self.previous_time = None

        self.last_receive_time = None

        self.initialized = False


    @staticmethod
    def low_pass(
        new_value,
        previous_value,
        cutoff_hz,
        dt
    ):

        if previous_value is None:
            return new_value

        if cutoff_hz <= 0.0:
            return new_value

        rc = 1.0 / (
            2.0
            * math.pi
            * cutoff_hz
        )

        alpha = dt / (
            rc + dt
        )

        return (
            previous_value
            + alpha
            * (
                new_value
                - previous_value
            )
        )


    def update(self, msg):

        if len(msg.data) <= 6:
            return False

        now = rospy.Time.now().to_sec()

        # ------------------------------------------------------------
        # Extract raw measurements
        # ------------------------------------------------------------

        velocity_raw = float(
            msg.data[6]
        )

        acceleration_raw = (
            float(msg.data[3])
            - self.imu_bias
        )

        self.velocity_raw = velocity_raw
        self.acceleration_raw = acceleration_raw

        # ------------------------------------------------------------
        # First measurement
        # ------------------------------------------------------------

        if self.previous_time is None:

            self.velocity = velocity_raw
            self.acceleration = acceleration_raw

            self.previous_time = now
            self.last_receive_time = now

            self.initialized = True

            return True

        # ------------------------------------------------------------
        # Actual callback interval
        # ------------------------------------------------------------

        dt = (
            now
            - self.previous_time
        )

        self.previous_time = now
        self.last_receive_time = now

        # The Arduino/publisher operates around 10 Hz.
        # Use 0.1 s if callback timing is clearly invalid.
        if dt <= 0.0 or dt > 0.5:
            dt = 0.1

        # ------------------------------------------------------------
        # Encoder 2 Hz LPF
        # ------------------------------------------------------------

        self.velocity = self.low_pass(
            velocity_raw,
            self.velocity,
            self.velocity_cutoff_hz,
            dt
        )

        # ------------------------------------------------------------
        # IMU 2 Hz LPF
        # ------------------------------------------------------------

        self.acceleration = self.low_pass(
            acceleration_raw,
            self.acceleration,
            self.acceleration_cutoff_hz,
            dt
        )

        self.initialized = True

        return True


# ====================================================================
# OCO MEASUREMENT BRIDGE
# ====================================================================

class ViconOCOMeasurements:

    def __init__(self):

        # ============================================================
        # Vehicle numbers
        # ============================================================

        self.leader_number = int(
            rospy.get_param(
                "~leader_number",
                1
            )
        )

        self.follower_number = int(
            rospy.get_param(
                "~follower_number",
                2
            )
        )

        # ============================================================
        # Vicon topics
        # ============================================================

        self.leader_vicon_topic = rospy.get_param(
            "~leader_vicon_topic",
            "/vicon/jetracer{}".format(
                self.leader_number
            )
        )

        self.follower_vicon_topic = rospy.get_param(
            "~follower_vicon_topic",
            "/vicon/jetracer{}".format(
                self.follower_number
            )
        )

        # ============================================================
        # Onboard sensor topics
        # ============================================================

        self.leader_sensor_topic = rospy.get_param(
            "~leader_sensor_topic",
            "/sensors_and_input_{}".format(
                self.leader_number
            )
        )

        self.follower_sensor_topic = rospy.get_param(
            "~follower_sensor_topic",
            "/sensors_and_input_{}".format(
                self.follower_number
            )
        )

        # ============================================================
        # Platoon parameters
        # ============================================================

        self.standstill_distance = float(
            rospy.get_param(
                "~standstill_distance",
                1.0
            )
        )

        self.headway = float(
            rospy.get_param(
                "~headway",
                0.5
            )
        )

        # ============================================================
        # Vicon filtering
        # ============================================================

        self.velocity_cutoff_hz = float(
            rospy.get_param(
                "~velocity_cutoff_hz",
                3.0
            )
        )

        self.acceleration_cutoff_hz = float(
            rospy.get_param(
                "~acceleration_cutoff_hz",
                2.0
            )
        )

        # ============================================================
        # Onboard filtering
        # ============================================================

        self.onboard_velocity_cutoff_hz = float(
            rospy.get_param(
                "~onboard_velocity_cutoff_hz",
                2.0
            )
        )

        self.onboard_acceleration_cutoff_hz = float(
            rospy.get_param(
                "~onboard_acceleration_cutoff_hz",
                2.0
            )
        )

        self.leader_imu_bias = float(
            rospy.get_param(
                "~leader_imu_bias",
                0.0
            )
        )

        self.follower_imu_bias = float(
            rospy.get_param(
                "~follower_imu_bias",
                0.0
            )
        )

        self.sensor_timeout = float(
            rospy.get_param(
                "~sensor_timeout",
                0.25
            )
        )

        # ============================================================
        # Existing Vicon publication rate
        # ============================================================

        self.publish_rate = float(
            rospy.get_param(
                "~publish_rate",
                10.0
            )
        )

        # ============================================================
        # Internal Vicon states
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
        # Internal onboard states
        # ============================================================

        self.leader_onboard = OnboardSensorState(
            self.onboard_velocity_cutoff_hz,
            self.onboard_acceleration_cutoff_hz,
            self.leader_imu_bias
        )

        self.follower_onboard = OnboardSensorState(
            self.onboard_velocity_cutoff_hz,
            self.onboard_acceleration_cutoff_hz,
            self.follower_imu_bias
        )

        # Vicon receive times used for validity/staleness checks
        self.last_leader_vicon_receive_time = None
        self.last_follower_vicon_receive_time = None

        # ============================================================
        # Existing Vicon diagnostic publishers
        # ============================================================

        self.leader_velocity_pub = rospy.Publisher(
            "/vicon_velocity_{}".format(
                self.leader_number
            ),
            Float32,
            queue_size=10
        )

        self.leader_acceleration_pub = rospy.Publisher(
            "/vicon_acceleration_{}".format(
                self.leader_number
            ),
            Float32,
            queue_size=10
        )

        self.follower_velocity_pub = rospy.Publisher(
            "/vicon_velocity_{}".format(
                self.follower_number
            ),
            Float32,
            queue_size=10
        )

        self.follower_acceleration_pub = rospy.Publisher(
            "/vicon_acceleration_{}".format(
                self.follower_number
            ),
            Float32,
            queue_size=10
        )

        self.spacing_error_pub = rospy.Publisher(
            "/spacing_error_meas_{}".format(
                self.follower_number
            ),
            Float32,
            queue_size=10
        )

        self.relative_velocity_pub = rospy.Publisher(
            "/relative_velocity_meas_{}".format(
                self.follower_number
            ),
            Float32,
            queue_size=10
        )

        self.distance_pub = rospy.Publisher(
            "/distance_meas_{}".format(
                self.follower_number
            ),
            Float32,
            queue_size=10
        )

        # ============================================================
        # Existing Vicon OCO measurement bank
        #
        # /oco/y1_3 ... /oco/y9_3
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
        # New onboard / hybrid OCO bank
        #
        # /oco_onboard/y1_3 ... /oco_onboard/y9_3
        # ============================================================

        self.onboard_publishers = {}

        for channel in range(1, 10):

            topic = "/oco_onboard/y{}_{}".format(
                channel,
                self.follower_number
            )

            self.onboard_publishers[channel] = rospy.Publisher(
                topic,
                Float32,
                queue_size=10
            )

        # ============================================================
        # Attack configuration
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

        self.attack_channels = [
            int(channel)
            for channel in self.attack_channels
        ]

        self.step_amplitude = float(
            rospy.get_param(
                "~step_amplitude",
                0.0
            )
        )

        self.white_noise_std = float(
            rospy.get_param(
                "~white_noise_std",
                0.0
            )
        )

        self.attack_start = float(
            rospy.get_param(
                "~attack_start",
                0.0
            )
        )

        self.attack_stop = float(
            rospy.get_param(
                "~attack_stop",
                -1.0
            )
        )

        self.switching_period = float(
            rospy.get_param(
                "~switching_period",
                4.0
            )
        )

        self.switching_on_time = float(
            rospy.get_param(
                "~switching_on_time",
                2.0
            )
        )

        self.start_time = time.monotonic()

        # ------------------------------------------------------------
        # Shared attack cache
        #
        # Both OCO banks should receive the same attack realization
        # within the same 10 Hz attack time slot.
        # ------------------------------------------------------------

        self.attack_cache_slot = None
        self.attack_cache = {}

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

        rospy.Subscriber(
            self.leader_sensor_topic,
            Float32MultiArray,
            self.leader_sensor_callback,
            queue_size=1
        )

        rospy.Subscriber(
            self.follower_sensor_topic,
            Float32MultiArray,
            self.follower_sensor_callback,
            queue_size=1
        )

        # ============================================================
        # Existing Vicon output remains timer-driven at 10 Hz
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0 / self.publish_rate
            ),
            self.timer_callback
        )

        # ============================================================
        # Information
        # ============================================================

        rospy.loginfo(
            "Vicon + onboard OCO measurement bridge started"
        )

        rospy.loginfo(
            "Leader Vicon: %s",
            self.leader_vicon_topic
        )

        rospy.loginfo(
            "Follower Vicon: %s",
            self.follower_vicon_topic
        )

        rospy.loginfo(
            "Leader onboard sensors: %s",
            self.leader_sensor_topic
        )

        rospy.loginfo(
            "Follower onboard sensors: %s",
            self.follower_sensor_topic
        )

        rospy.loginfo(
            "Vicon velocity LPF: %.2f Hz",
            self.velocity_cutoff_hz
        )

        rospy.loginfo(
            "Vicon acceleration LPF: %.2f Hz",
            self.acceleration_cutoff_hz
        )

        rospy.loginfo(
            "Onboard encoder LPF: %.2f Hz",
            self.onboard_velocity_cutoff_hz
        )

        rospy.loginfo(
            "Onboard IMU LPF: %.2f Hz",
            self.onboard_acceleration_cutoff_hz
        )

        rospy.loginfo(
            "Leader IMU bias: %.3f m/s^2",
            self.leader_imu_bias
        )

        rospy.loginfo(
            "Follower IMU bias: %.3f m/s^2",
            self.follower_imu_bias
        )

        rospy.loginfo(
            "Sensor timeout: %.3f s",
            self.sensor_timeout
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
    # VICON CALLBACKS
    # ================================================================

    def leader_vicon_callback(self, msg):

        self.leader.update(msg)

        self.last_leader_vicon_receive_time = (
            rospy.Time.now().to_sec()
        )


    def follower_vicon_callback(self, msg):

        self.follower.update(msg)

        self.last_follower_vicon_receive_time = (
            rospy.Time.now().to_sec()
        )


    # ================================================================
    # ONBOARD SENSOR CALLBACKS
    # ================================================================

    def leader_sensor_callback(self, msg):

        success = self.leader_onboard.update(
            msg
        )

        if not success:

            rospy.logwarn_throttle(
                2.0,
                "Leader /sensors_and_input_%d has insufficient data",
                self.leader_number
            )


    def follower_sensor_callback(self, msg):
        """
        Follower sensor arrival is the trigger for the onboard/hybrid
        OCO measurement bank.

        This means there is no additional 10 Hz timer delay between
        receiving follower data and publishing /oco_onboard/y1...y9.
        """

        success = self.follower_onboard.update(
            msg
        )

        if not success:

            rospy.logwarn_throttle(
                2.0,
                "Follower /sensors_and_input_%d has insufficient data",
                self.follower_number
            )

            return

        # Immediately attempt publication.
        # If leader/Vicon data is not available yet, this simply returns.
        self.publish_onboard_measurements()


    # ================================================================
    # ATTACK FUNCTIONS
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

        attack_elapsed = (
            elapsed
            - self.attack_start
        )

        phase = (
            attack_elapsed
            % self.switching_period
        )

        return (
            phase
            < self.switching_on_time
        )


    def attack_offset(self, channel, elapsed):
        """
        Return only the attack contribution.

        The contribution is cached for one 10 Hz attack slot so the
        Vicon bank and onboard bank receive the SAME random attack
        realization when using white-noise attacks.
        """

        if not self.attack_enabled:
            return 0.0

        if channel not in self.attack_channels:
            return 0.0

        if not self.attack_time_active(elapsed):
            return 0.0

        attack_type = self.attack_type.lower()

        if attack_type == "none":
            return 0.0

        if attack_type == "switching_white_noise":

            if not self.switching_active(elapsed):
                return 0.0

        if attack_type == "switching_step":

            if not self.switching_active(elapsed):
                return 0.0

        # ------------------------------------------------------------
        # Step attacks are deterministic
        # ------------------------------------------------------------

        if attack_type == "step":
            return self.step_amplitude

        if attack_type == "switching_step":
            return self.step_amplitude

        # ------------------------------------------------------------
        # White-noise attacks:
        # one realization per channel per 10 Hz time slot
        # ------------------------------------------------------------

        if attack_type in [
            "white_noise",
            "switching_white_noise"
        ]:

            slot = int(
                elapsed
                * self.publish_rate
            )

            if slot != self.attack_cache_slot:

                self.attack_cache_slot = slot
                self.attack_cache = {}

            if channel not in self.attack_cache:

                self.attack_cache[channel] = (
                    random.gauss(
                        0.0,
                        self.white_noise_std
                    )
                )

            return self.attack_cache[channel]

        rospy.logwarn_throttle(
            5.0,
            "Unknown attack type '%s'; publishing clean data.",
            self.attack_type
        )

        return 0.0


    def apply_attack(
        self,
        channel,
        clean_value,
        elapsed
    ):

        return (
            clean_value
            + self.attack_offset(
                channel,
                elapsed
            )
        )


    # ================================================================
    # HELPERS
    # ================================================================

    @staticmethod
    def publish_float(
        publisher,
        value
    ):

        msg = Float32()

        msg.data = float(
            value
        )

        publisher.publish(
            msg
        )


    def get_vicon_distance(self):

        if self.leader.x is None:
            return None

        if self.leader.y is None:
            return None

        if self.follower.x is None:
            return None

        if self.follower.y is None:
            return None

        dx = (
            self.leader.x
            - self.follower.x
        )

        dy = (
            self.leader.y
            - self.follower.y
        )

        return math.sqrt(
            dx * dx
            + dy * dy
        )


    def onboard_inputs_valid(self):

        now = rospy.Time.now().to_sec()

        # ------------------------------------------------------------
        # Leader onboard data
        # ------------------------------------------------------------

        if not self.leader_onboard.initialized:

            rospy.logwarn_throttle(
                2.0,
                "Waiting for leader onboard sensor data"
            )

            return False

        if self.leader_onboard.last_receive_time is None:
            return False

        leader_age = (
            now
            - self.leader_onboard.last_receive_time
        )

        if leader_age > self.sensor_timeout:

            rospy.logwarn_throttle(
                2.0,
                "Leader onboard sensor data stale: %.3f s",
                leader_age
            )

            return False

        # ------------------------------------------------------------
        # Follower onboard data
        # ------------------------------------------------------------

        if not self.follower_onboard.initialized:
            return False

        if self.follower_onboard.last_receive_time is None:
            return False

        follower_age = (
            now
            - self.follower_onboard.last_receive_time
        )

        if follower_age > self.sensor_timeout:

            rospy.logwarn_throttle(
                2.0,
                "Follower onboard sensor data stale: %.3f s",
                follower_age
            )

            return False

        # ------------------------------------------------------------
        # Vicon distance
        # ------------------------------------------------------------

        if self.get_vicon_distance() is None:

            rospy.logwarn_throttle(
                2.0,
                "Waiting for Vicon position for onboard spacing measurement"
            )

            return False

        if self.last_leader_vicon_receive_time is None:
            return False

        if self.last_follower_vicon_receive_time is None:
            return False

        leader_vicon_age = (
            now
            - self.last_leader_vicon_receive_time
        )

        follower_vicon_age = (
            now
            - self.last_follower_vicon_receive_time
        )

        if leader_vicon_age > self.sensor_timeout:

            rospy.logwarn_throttle(
                2.0,
                "Leader Vicon data stale for distance: %.3f s",
                leader_vicon_age
            )

            return False

        if follower_vicon_age > self.sensor_timeout:

            rospy.logwarn_throttle(
                2.0,
                "Follower Vicon data stale for distance: %.3f s",
                follower_vicon_age
            )

            return False

        return True


    # ================================================================
    # NEW ONBOARD/HYBRID OCO OUTPUT
    # ================================================================

    def publish_onboard_measurements(self):

        if not self.onboard_inputs_valid():
            return

        # ============================================================
        # FILTERED ONBOARD LONGITUDINAL MEASUREMENTS
        # ============================================================

        v_leader = (
            self.leader_onboard.velocity
        )

        a_leader = (
            self.leader_onboard.acceleration
        )

        v_follower = (
            self.follower_onboard.velocity
        )

        a_follower = (
            self.follower_onboard.acceleration
        )

        # ============================================================
        # DISTANCE REMAINS VICON-BASED FOR NOW
        # ============================================================

        distance = (
            self.get_vicon_distance()
        )

        # ============================================================
        # ONBOARD/HYBRID SPACING ERROR
        #
        # e = d_vicon - s - h * v_follower_encoder
        # ============================================================

        spacing_error = (
            distance
            - self.standstill_distance
            - self.headway
            * v_follower
        )

        # ============================================================
        # RELATIVE VELOCITY FROM ONBOARD ENCODERS
        # ============================================================

        delta_v = (
            v_leader
            - v_follower
        )

        # ============================================================
        # NINE-CHANNEL ONBOARD/HYBRID BANK
        # ============================================================

        clean_measurements = {

            1: spacing_error,

            2: v_follower,

            3: a_follower,

            4: delta_v,

            5: a_leader,

            6: spacing_error,

            7: v_follower,

            8: spacing_error,

            9: v_follower,
        }

        elapsed = (
            time.monotonic()
            - self.start_time
        )

        # ============================================================
        # ATTACK + PUBLISH
        # ============================================================

        for channel, clean_value in clean_measurements.items():

            output_value = self.apply_attack(
                channel,
                clean_value,
                elapsed
            )

            self.publish_float(
                self.onboard_publishers[channel],
                output_value
            )


    # ================================================================
    # EXISTING VICON 10 HZ OUTPUT
    # ================================================================

    def timer_callback(self, event):

        # Existing Vicon bank waits until velocity and acceleration
        # have been initialized for both cars.
        if not self.leader.initialized:
            return

        if not self.follower.initialized:
            return

        # ============================================================
        # Extract Vicon-derived physical quantities
        # ============================================================

        v1 = self.leader.velocity
        a1 = self.leader.acceleration

        v2 = self.follower.velocity
        a2 = self.follower.acceleration

        # ============================================================
        # Relative velocity
        # ============================================================

        delta_v = (
            v1
            - v2
        )

        # ============================================================
        # Vicon distance
        # ============================================================

        distance = (
            self.get_vicon_distance()
        )

        if distance is None:
            return

        # ============================================================
        # Constant-time-headway spacing error
        # ============================================================

        spacing_error = (
            distance
            - self.standstill_distance
            - self.headway
            * v2
        )

        # ============================================================
        # Existing clean physical Vicon measurements
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
        # Existing Vicon nine-channel OCO bank
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

        elapsed = (
            time.monotonic()
            - self.start_time
        )

        # ============================================================
        # Attack + publish existing Vicon bank
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


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":

    rospy.init_node(
        "vicon_oco_measurements"
    )

    node = ViconOCOMeasurements()

    rospy.spin()
