#!/usr/bin/env python3

import rospy
import numpy as np

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray


class OCONineObservers:
    def __init__(self):

        # ============================================================
        # General parameters
        # ============================================================

        self.update_rate = rospy.get_param("~update_rate", 10.0)

        # Optional debugging:
        # Keep individual state publishing available, but disabled.
        self.publish_individual_states = rospy.get_param(
            "~publish_individual_states",
            False
        )

        # ============================================================
        # OCO classification parameters
        # ============================================================

        # Number of observers
        self.N = 9

        # Paper:
        #
        # beta_bar_eta = 1 - 1/N
        #
        self.beta_bar_eta = 1.0 - 1.0 / self.N

        # Magnifier used by the paper
        self.a_beta = float(
	    rospy.get_param(
            	"~a_beta",
            	1000.0
            )
	)

        # Noise bounds.
        #
        # For the current clean simulation we use zero.
        #
        self.Bw = float(
	    rospy.get_param(
            	"~Bw",
            	0.0
	    )
        )

        self.Bgamma = float(
	    rospy.get_param(
            	"~Bgamma",
            	0.0
	    )
        )

        # Numerical protection against division by zero
        self.beta_epsilon = float(
	    rospy.get_param(
            	"~beta_epsilon",
            	1e-12
	    )
        )

        # ============================================================
        # Measurement topics
        # ============================================================

        self.y_topics = {
            1: rospy.get_param("~y1_topic", "/car2/oco/y1"),
            2: rospy.get_param("~y2_topic", "/car2/oco/y2"),
            6: rospy.get_param("~y6_topic", "/car2/oco/y6"),
            7: rospy.get_param("~y7_topic", "/car2/oco/y7"),
            8: rospy.get_param("~y8_topic", "/car2/oco/y8"),
            9: rospy.get_param("~y9_topic", "/car2/oco/y9"),
        }

        # ============================================================
        # Known model input topics
        # ============================================================

        self.leader_acc_topic = rospy.get_param(
            "~leader_acc_topic",
            "/car1/acc_saturated_1"
        )

        self.follower_acc_topic = rospy.get_param(
            "~follower_acc_topic",
            "/car2/acc_saturated_1"
        )

        # ============================================================
        # Discrete vehicle model
        #
        # State:
        #
        # x = [
        #     e,
        #     v2,
        #     a2,
        #     delta_v,
        #     a1
        # ]
        #
        # Ts  = 0.1 s
        # h   = 0.5 s
        # tau = 0.1 s
        # ============================================================

        self.A = np.array([
            [1.0, 0.0, -0.035321205588, 0.1, 0.003678794412],
            [0.0, 1.0,  0.063212055883, 0.0, 0.0],
            [0.0, 0.0,  0.367879441171, 0.0, 0.0],
            [0.0, 0.0, -0.063212055883, 1.0, 0.063212055883],
            [0.0, 0.0,  0.0, 0.0, 0.367879441171]
        ], dtype=float)

        self.B1 = np.array([
            0.001321205588,
            0.0,
            0.0,
            0.036787944117,
            0.632120558829
        ], dtype=float)

        self.B2 = np.array([
            -0.019660602794,
             0.036787944117,
             0.632120558829,
            -0.036787944117,
             0.0
        ], dtype=float)

        # ============================================================
        # Measurement matrices
        # ============================================================

        # Measurement order [e, v2]
        self.C_ev = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0]
        ], dtype=float)

        # Measurement order [v2, e]
        self.C_ve = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=float)

        # ============================================================
        # LMI-designed observer gains
        # ============================================================

        # Measurement order [e, v2]
        self.L_ev = np.array([
            [ 0.520908960268, -0.000895941310526],
            [-0.000307673411836,  0.500868994999],
            [-0.00298624749008,   0.00519347482535],
            [ 0.106763039631,    -0.00226423486617],
            [ 0.000510883243397,  0.00005409863919]
        ], dtype=float)

        # Measurement order [v2, e]
        self.L_ve = np.array([
            [-0.000895941310526,  0.520908960268],
            [ 0.500868994999,    -0.000307673411836],
            [ 0.00519347482535,  -0.00298624749008],
            [-0.00226423486617,   0.106763039631],
            [ 0.00005409863919,   0.000510883243397]
        ], dtype=float)

        # ============================================================
        # Nine observer subsets
        #
        # J1 = {1,2}
        # J2 = {1,7}
        # J3 = {1,9}
        # J4 = {2,6}
        # J5 = {2,8}
        # J6 = {6,7}
        # J7 = {6,9}
        # J8 = {7,8}
        # J9 = {8,9}
        # ============================================================

        self.observer_config = {
            1: ([1, 2], self.C_ev, self.L_ev),
            2: ([1, 7], self.C_ev, self.L_ev),
            3: ([1, 9], self.C_ev, self.L_ev),

            4: ([2, 6], self.C_ve, self.L_ve),
            5: ([2, 8], self.C_ve, self.L_ve),

            6: ([6, 7], self.C_ev, self.L_ev),
            7: ([6, 9], self.C_ev, self.L_ev),

            8: ([7, 8], self.C_ve, self.L_ve),

            9: ([8, 9], self.C_ev, self.L_ev),
        }

        # ============================================================
        # Nine vehicle observer states
        # ============================================================

        self.xhat = {}

        for j in range(1, 10):
            self.xhat[j] = np.zeros(5, dtype=float)

        # ============================================================
        # Residual-reference model
        #
        # xr_j(k+1) =
        # Ar*xr_j(k) + Br*||r_j(k)||_2
        #
        # eta_j(k) = xr_j,1(k)
        #
        # Kr = 2
        # Cr = 3
        # Ts = 0.1
        #
        # Exact ZOH discretization
        # ============================================================

        self.Ar = np.array([
            [ 0.990944082994,  0.086106664957],
            [-0.172213329915,  0.732624088120]
        ], dtype=float)

        self.Br = np.array([
            0.009055917006,
            0.172213329915
        ], dtype=float)

        # Nine residual-reference states
        self.xr = {}

        for j in range(1, 10):
            self.xr[j] = np.zeros(2, dtype=float)

        # ============================================================
        # Eta and beta storage
        # ============================================================

        self.eta = np.zeros(9, dtype=float)

        self.beta_eta = np.full(
            9,
            self.beta_bar_eta,
            dtype=float
        )

        # Paper specifies beta_j(0) in (0,1).
        # Clean/equal case gives beta = 0.5.
        self.beta = np.full(
            9,
            0.5,
            dtype=float
        )

        # ============================================================
        # Latest measurements and inputs
        # ============================================================

        self.y = {
            1: None,
            2: None,
            6: None,
            7: None,
            8: None,
            9: None
        }

        self.u1 = None
        self.u2 = None

        # ============================================================
        # Subscribers
        # ============================================================

        for channel in [1, 2, 6, 7, 8, 9]:

            rospy.Subscriber(
                self.y_topics[channel],
                Float32,
                self.measurement_callback,
                callback_args=channel,
                queue_size=10
            )

        rospy.Subscriber(
            self.leader_acc_topic,
            Float32,
            self.leader_acc_callback,
            queue_size=10
        )

        rospy.Subscriber(
            self.follower_acc_topic,
            Float32,
            self.follower_acc_callback,
            queue_size=10
        )

        # ============================================================
        # Optional individual observer state publishers
        # ============================================================

        self.state_publishers = {}

        if self.publish_individual_states:

            for j in range(1, 10):

                topic = (
                    "/car2/oco/observer_"
                    + str(j)
                    + "/state_estimate"
                )

                self.state_publishers[j] = rospy.Publisher(
                    topic,
                    Float32MultiArray,
                    queue_size=10
                )

        # ============================================================
        # Eta publisher
        #
        # [eta1, eta2, ..., eta9]
        # ============================================================

        self.eta_publisher = rospy.Publisher(
            "/car2/oco/eta",
            Float32MultiArray,
            queue_size=10
        )

        # ============================================================
        # Beta publisher
        #
        # [beta1, beta2, ..., beta9]
        # ============================================================

        self.beta_publisher = rospy.Publisher(
            "/car2/oco/beta",
            Float32MultiArray,
            queue_size=10
        )

        # ============================================================
        # Timer
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.update_rate),
            self.update_callback
        )

        # ============================================================
        # Startup logging
        # ============================================================

        rospy.loginfo("========================================")
        rospy.loginfo("Nine OCO observers started")
        rospy.loginfo("Residual reference models enabled")
        rospy.loginfo("Classification beta enabled")
        rospy.loginfo("========================================")

        rospy.loginfo(
            "beta_bar_eta = %.9f",
            self.beta_bar_eta
        )

        rospy.loginfo(
            "a_beta = %.3f",
            self.a_beta
        )

        rospy.loginfo(
            "Bw = %.6g, Bgamma = %.6g",
            self.Bw,
            self.Bgamma
        )

        rospy.loginfo(
            "Publish individual states: %s",
            self.publish_individual_states
        )

    # ================================================================
    # Callbacks
    # ================================================================

    def measurement_callback(self, msg, channel):

        self.y[channel] = float(msg.data)

    def leader_acc_callback(self, msg):

        self.u1 = float(msg.data)

    def follower_acc_callback(self, msg):

        self.u2 = float(msg.data)

    # ================================================================
    # Data-ready check
    # ================================================================

    def data_ready(self):

        if self.u1 is None:
            return False

        if self.u2 is None:
            return False

        for channel in [1, 2, 6, 7, 8, 9]:

            if self.y[channel] is None:
                return False

        return True

    # ================================================================
    # Calculate classification ratios
    # ================================================================

    def calculate_beta(self):

        # ------------------------------------------------------------
        # Paper:
        #
        # beta_eta_j =
        #
        # 1 -
        #
        # (eta_j + Bw + Bgamma)
        # -----------------------------------
        # sum_s(eta_s + Bw + Bgamma)
        #
        # ------------------------------------------------------------

        eta_with_bounds = (
            self.eta
            + self.Bw
            + self.Bgamma
        )

        denominator = np.sum(
            eta_with_bounds
        )

        # ------------------------------------------------------------
        # Startup / perfectly clean numerical safeguard
        #
        # If all eta values and both noise bounds are zero,
        # denominator = 0.
        #
        # In the equal clean condition:
        #
        # beta_eta = 1 - 1/N = 8/9
        #
        # which gives beta = 0.5.
        # ------------------------------------------------------------

        if denominator <= self.beta_epsilon:

            self.beta_eta[:] = self.beta_bar_eta

        else:

            self.beta_eta = (
                1.0
                - eta_with_bounds / denominator
            )

        # ------------------------------------------------------------
        # Paper's nonlinear activation:
        #
        # beta_j =
        #
        # 1/pi *
        # atan(
        #     a_beta *
        #     (beta_eta_j - beta_bar_eta)
        # )
        # + 0.5
        #
        # ------------------------------------------------------------

        self.beta = (
            (1.0 / np.pi)
            * np.arctan(
                self.a_beta
                * (
                    self.beta_eta
                    - self.beta_bar_eta
                )
            )
            + 0.5
        )

    # ================================================================
    # Main update
    # ================================================================

    def update_callback(self, event):

        if not self.data_ready():
            return

        # ============================================================
        # STEP 1
        #
        # Current eta values are taken from the CURRENT
        # residual-reference model states xr_j(k).
        #
        # These eta values are then used to calculate beta(k).
        # ============================================================

        for j in range(1, 10):

            self.eta[j - 1] = self.xr[j][0]

        # ============================================================
        # STEP 2
        #
        # Calculate beta_eta(k) and beta(k)
        # ============================================================

        self.calculate_beta()

        # ============================================================
        # STEP 3
        #
        # Calculate residuals and update all nine vehicle observers.
        #
        # IMPORTANT:
        # These are STILL INDEPENDENT observers.
        #
        # The coupling term is NOT added yet.
        # ============================================================

        for j in range(1, 10):

            channels, C, L = self.observer_config[j]

            # Measurement vector
            yj = np.array([
                self.y[channels[0]],
                self.y[channels[1]]
            ], dtype=float)

            xhat_current = self.xhat[j]

            # --------------------------------------------------------
            # Residual
            #
            # r_j(k) = y_j(k) - C_j*xhat_j(k)
            # --------------------------------------------------------

            residual = (
                yj
                - C.dot(xhat_current)
            )

            # --------------------------------------------------------
            # Independent observer update
            #
            # xhat_j(k+1) =
            #
            # A*xhat_j(k)
            # + B1*u1(k)
            # + B2*u2(k)
            # + L_j*r_j(k)
            # --------------------------------------------------------

            xhat_next = (
                self.A.dot(xhat_current)
                + self.B1 * self.u1
                + self.B2 * self.u2
                + L.dot(residual)
            )

            self.xhat[j] = xhat_next

            # --------------------------------------------------------
            # Residual magnitude
            #
            # ||r_j(k)||_2
            # --------------------------------------------------------

            residual_magnitude = np.linalg.norm(
                residual,
                ord=2
            )

            # --------------------------------------------------------
            # Update residual-reference model:
            #
            # xr_j(k+1) =
            #
            # Ar*xr_j(k)
            # + Br*||r_j(k)||_2
            # --------------------------------------------------------

            xr_next = (
                self.Ar.dot(self.xr[j])
                + self.Br * residual_magnitude
            )

            self.xr[j] = xr_next

        # ============================================================
        # STEP 4
        #
        # Publish eta(k)
        # ============================================================

        eta_msg = Float32MultiArray()

        eta_msg.data = self.eta.tolist()

        self.eta_publisher.publish(
            eta_msg
        )

        # ============================================================
        # STEP 5
        #
        # Publish beta(k)
        # ============================================================

        beta_msg = Float32MultiArray()

        beta_msg.data = self.beta.tolist()

        self.beta_publisher.publish(
            beta_msg
        )

        # ============================================================
        # OPTIONAL DEBUGGING
        #
        # Publish individual observer estimates.
        # ============================================================

        if self.publish_individual_states:

            for j in range(1, 10):

                state_msg = Float32MultiArray()

                state_msg.data = self.xhat[j].tolist()

                self.state_publishers[j].publish(
                    state_msg
                )


if __name__ == "__main__":

    rospy.init_node("oco_nine_observers")

    node = OCONineObservers()

    rospy.spin()
