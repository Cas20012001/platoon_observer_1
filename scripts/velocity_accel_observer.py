#!/usr/bin/env python3

import math
import rospy
from std_msgs.msg import Float32, Float32MultiArray, Float64MultiArray


class VelocityAccelerationObserver:
    """
    Observer state:
        x_hat = [v_hat, a_hat]^T

    Model:
        x_hat[k+1] = Ad*x_hat[k] + Bd*u[k] + Ld*(y[k] - Cd*x_hat[k])

    Measurements:
        y = [encoder velocity, IMU acc_x]^T

    For teleop:
        u is estimated from throttle using DART throttle-to-acceleration model.

    Later:
        u can be taken directly from /desired_acc_<car_number> or /acc_saturated_<car_number>.
    """

    def __init__(self):
        self.car_number = int(rospy.get_param("~car_number", 3))

        # Topics
        self.combined_topic = rospy.get_param(
            "~combined_topic",
            "/sensors_and_input_{}".format(self.car_number)
        )
        self.desired_acc_topic = rospy.get_param(
            "~desired_acc_topic",
            "/desired_acc_{}".format(self.car_number)
        )
        self.acc_saturated_topic = rospy.get_param(
            "~acc_saturated_topic",
            "/acc_saturated_{}".format(self.car_number)
        )
        self.state_estimate_topic = rospy.get_param("~state_estimate_topic", "/state_estimate")
        self.observer_input_topic = rospy.get_param("~observer_input_topic", "/observer_input_acc")

        # Input mode:
        #   "throttle_model" -> use throttle + velocity model, good for teleop testing
        #   "desired_acc"    -> use /desired_acc_3 directly
        #   "acc_saturated"  -> use /acc_saturated_3 directly
        self.input_mode = rospy.get_param("~input_mode", "throttle_model")

        # Indices in /sensors_and_input_3
        self.velocity_index = int(rospy.get_param("~velocity_index", 6))
        self.acceleration_index = int(rospy.get_param("~acceleration_index", 3))
        self.throttle_index = int(rospy.get_param("~throttle_index", 8))

        # Signal corrections
        self.velocity_sign = float(rospy.get_param("~velocity_sign", 1.0))
        self.accel_sign = float(rospy.get_param("~accel_sign", 1.0))
        self.throttle_sign = float(rospy.get_param("~throttle_sign", 1.0))
        self.imu_accel_bias = float(rospy.get_param("~imu_accel_bias", 0.0))

        # Timing
        self.sample_time = float(rospy.get_param("~sample_time", 0.1))

        # Observer matrices for Ts=0.1, tau=0.1 example
        self.Ad = [
            [float(rospy.get_param("~Ad_00", 1.0000)), float(rospy.get_param("~Ad_01", 0.0632))],
            [float(rospy.get_param("~Ad_10", 0.0000)), float(rospy.get_param("~Ad_11", 0.3679))]
        ]

        self.Bd = [
            float(rospy.get_param("~Bd_0", 0.0368)),
            float(rospy.get_param("~Bd_1", 0.6321))
        ]

        self.Ld = [
            [float(rospy.get_param("~Ld_00", 0.6321)), float(rospy.get_param("~Ld_01", 0.0632))],
            [float(rospy.get_param("~Ld_10", 0.0000)), float(rospy.get_param("~Ld_11", 0.2325))]
        ]

        # State estimate
        self.v_hat = float(rospy.get_param("~initial_velocity", 0.0))
        self.a_hat = float(rospy.get_param("~initial_acceleration", 0.0))

        # Latest measurements/input
        self.v_meas = 0.0
        self.a_meas = 0.0
        self.throttle = 0.0
        self.desired_acc = 0.0
        self.acc_saturated = 0.0
        self.have_sensor_msg = False

        # DART longitudinal model parameters
        self.mass = float(rospy.get_param("~mass", 1.67))

        # Optional scale factor, useful if model is too aggressive/weak on your robot
        self.model_accel_scale = float(rospy.get_param("~model_accel_scale", 1.0))

        # Publishers
        self.state_pub = rospy.Publisher(
            self.state_estimate_topic,
            Float64MultiArray,
            queue_size=1
        )
        self.input_pub = rospy.Publisher(
            self.observer_input_topic,
            Float32,
            queue_size=1
        )

        # Subscribers
        self.sensors_sub = rospy.Subscriber(
            self.combined_topic,
            Float32MultiArray,
            self.sensors_callback,
            queue_size=1
        )
        self.desired_acc_sub = rospy.Subscriber(
            self.desired_acc_topic,
            Float32,
            self.desired_acc_callback,
            queue_size=1
        )
        self.acc_saturated_sub = rospy.Subscriber(
            self.acc_saturated_topic,
            Float32,
            self.acc_saturated_callback,
            queue_size=1
        )

        self.timer = rospy.Timer(rospy.Duration(self.sample_time), self.update)

        rospy.loginfo("velocity_accel_observer started")
        rospy.loginfo("combined_topic: %s", self.combined_topic)
        rospy.loginfo("input_mode: %s", self.input_mode)
        rospy.loginfo("velocity_index=%d, acceleration_index=%d, throttle_index=%d",
                      self.velocity_index, self.acceleration_index, self.throttle_index)

    def sensors_callback(self, msg):
        data = msg.data

        max_needed_index = max(self.velocity_index, self.acceleration_index, self.throttle_index)
        if len(data) <= max_needed_index:
            rospy.logwarn_throttle(
                2.0,
                "Received %s with length %d, but need index %d",
                self.combined_topic,
                len(data),
                max_needed_index
            )
            return

        self.v_meas = self.velocity_sign * float(data[self.velocity_index])
        self.a_meas = self.accel_sign * (float(data[self.acceleration_index]) - self.imu_accel_bias)
        self.throttle = self.throttle_sign * float(data[self.throttle_index])
        self.have_sensor_msg = True

    def desired_acc_callback(self, msg):
        self.desired_acc = float(msg.data)

    def acc_saturated_callback(self, msg):
        self.acc_saturated = float(msg.data)

    def get_motor_params(self):
        """
        Parameters copied from the DART acc_2_throttle implementation.

        motor_force(th, v):
            Fm = (a_m - v*b_m) * w * (th + c_m)
            w = 0.5 * (tanh(100*(th+c_m)) + 1)
        """
        if self.car_number == 2:
            return 26.47014617919922, 8.640666007995605, -0.1981888711452484
        elif self.car_number == 4:
            return 28.323787689208984, 8.21423053741455, -0.13714951276779175
        else:
            # Car 1 and car 3
            return 28.887779235839844, 5.986172199249268, -0.15045104920864105

    def get_friction_params(self):
        """
        Parameters copied from the DART acc_2_throttle implementation.

        friction(v):
            Ff = -a_f*tanh(b_f*v) - c_f*v
        """
        if self.car_number == 2:
            return 1.6498010158538818, 15.262519836425781, 0.009999999776482582
        elif self.car_number == 4:
            return 1.767649531364441, 13.065838813781738, 0.009999999776482582
        else:
            # Car 1 and car 3
            return 1.7194761037826538, 13.312559127807617, 0.289848655462265

    def motor_force(self, throttle, velocity):
        a_m, b_m, c_m = self.get_motor_params()

        w = 0.5 * (math.tanh(100.0 * (throttle + c_m)) + 1.0)
        fm = (a_m - velocity * b_m) * w * (throttle + c_m)

        return fm

    def friction_force(self, velocity):
        a_f, b_f, c_f = self.get_friction_params()

        ff = -a_f * math.tanh(b_f * velocity) - velocity * c_f

        return ff

    def acceleration_from_throttle_model(self):
        """
        Converts teleop throttle and encoder velocity into approximate acceleration.

        This is for observer testing while teleop publishes throttle.
        Later, use input_mode="desired_acc" or "acc_saturated".
        """
        fm = self.motor_force(self.throttle, self.v_meas)
        ff = self.friction_force(self.v_meas)

        acc = (fm + ff) / self.mass
        acc *= self.model_accel_scale

        return acc

    def get_input_acceleration(self):
        if self.input_mode == "desired_acc":
            return self.desired_acc

        if self.input_mode == "acc_saturated":
            return self.acc_saturated

        if self.input_mode == "throttle_model":
            return self.acceleration_from_throttle_model()

        rospy.logwarn_throttle(
            2.0,
            "Unknown input_mode '%s'. Falling back to throttle_model.",
            self.input_mode
        )
        return self.acceleration_from_throttle_model()

    def update(self, event):
        if not self.have_sensor_msg:
            rospy.logwarn_throttle(2.0, "Waiting for sensor message on %s", self.combined_topic)
            return

        u = self.get_input_acceleration()

        # y - C*x_hat, with C = I
        innovation_v = self.v_meas - self.v_hat
        innovation_a = self.a_meas - self.a_hat

        v_next = (
            self.Ad[0][0] * self.v_hat
            + self.Ad[0][1] * self.a_hat
            + self.Bd[0] * u
            + self.Ld[0][0] * innovation_v
            + self.Ld[0][1] * innovation_a
        )

        a_next = (
            self.Ad[1][0] * self.v_hat
            + self.Ad[1][1] * self.a_hat
            + self.Bd[1] * u
            + self.Ld[1][0] * innovation_v
            + self.Ld[1][1] * innovation_a
        )

        self.v_hat = v_next
        self.a_hat = a_next

        state_msg = Float64MultiArray()
        state_msg.data = [
            self.v_hat,
            self.a_hat,
            self.v_meas,
            self.a_meas,
            u,
            self.throttle
        ]
        self.state_pub.publish(state_msg)

        input_msg = Float32()
        input_msg.data = float(u)
        self.input_pub.publish(input_msg)


if __name__ == "__main__":
    rospy.init_node("velocity_accel_observer")
    node = VelocityAccelerationObserver()
    rospy.spin()
