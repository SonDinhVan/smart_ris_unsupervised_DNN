"""
This class provides methods to manipulate the network topology, generate
parameters and calculate performances such as ergodic spectral efficiency,
its upper bound, ... (both analytics and simulations).
"""

from numpy.linalg.linalg import _matrix_rank_dispatcher
from module import node
from module import channel as cl

import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import math
import itertools
import scipy.special as ss
from dataclasses import dataclass
import time


def marcum_q(M: np.float, a: np.float, b: np.float) -> np.float:
    """
    Calculate Marcum-Q function. For more information, visit link
    https://en.wikipedia.org/wiki/Marcum_Q-function
    We calculate using the sum of the first 10 values of the sequences,
    which guarantees a satisfactory accuration.
    This function is used for calculating the closed-form expression of
    outage probability.

    Args:
        M, a, b: np.float - Parameters to calculate the Q-function.
    """
    # The marcum Q function, calculated by summing its 10 components.
    res = sum(
        (a ** 2 / 2) ** k * ss.gammainc(M + k, b ** 2 / 2) / math.factorial(k)
        for k in range(10)
    )
    return 1.0 - np.exp(-(a ** 2) / 2) * res


@dataclass
class LargeScaleCoef:
    """
    F = [f_1, f_2, ..., f_k]^T is column vector
    R = [r_1, r_2, ..., r_k]^T is column vector
    A = [AoA_array_1_product, AoA_array_2_product, ..., AoA_array_K_product]^T
        with angle_i is the i-th column vector for user k, which is equal to
        the element wise product between conjugate AoA_array between CH_i-RIS
        and RIS-BS.
    cascaded_AoA = [sin(omega_i) - sin(omega_0), ...]^T. This can be
        used to recreate the angle matrix A. The cascaded_angle is used since
        it allows us to save the neccessary information for training without
        using too much RAM.
    For more information, refer to the paper
    """

    F: np.array = None
    R: np.array = None
    A: np.array = None
    cascaded_AoA: np.array = None


class Network:
    """
    Class contains one configuration of nodes in the network.
    """

    def __init__(self, config: Dict) -> None:
        """
        Load the parameters from the config and construct BS, RIS object.
        Note that CH_list are not touched yet since the locations of CHs
        are going to be distributed, which will be handled in the next
        function, i.e. generate_one_CH_realizations()

        Args:
            config (Dict): [The configuration Dict]
        """
        # Configuration Dict imported
        self.config = config
        # noise power
        self.noise_power = 10 ** (self.config["channel"]["noise_power"] / 10)
        # Load the configuration
        P = 10 ** (self.config["system_model"]["BS"]["P"] / 10) / 1000
        x_bs = self.config["system_model"]["BS"]["x_bs"]
        y_bs = self.config["system_model"]["BS"]["y_bs"]
        num_phase = self.config["system_model"]["RIS"]["num_phase"]
        x_ris = self.config["system_model"]["RIS"]["x_ris"]
        y_ris = self.config["system_model"]["RIS"]["y_ris"]

        # Class attributes
        # Create a BS at the given location
        self.BS = node.BaseStation(x=x_bs, y=y_bs, power=P)
        # Create a RIS at the given location
        self.RIS = node.ReIntelSurface(x=x_ris, y=y_ris, num_phase=num_phase)
        # Number of Cluster Heads
        self.K = self.config["system_model"]["CH"]["K"]
        # List of CHs
        self.CH_list = None

    def generate_one_CHs_realization(self, r_max: np.float) -> None:
        """
        Generate one CHs realization with the locations of CHs are
        randomly distributed. All the parameters related to channel will
        also be assigned to object RIS, BS and CHs.

        Since each CH is responsible for a small area in the factory, each
        will be distributed in an arc of the circle with radius of r_max.
        The more the number of CHs, the smaller area in the circle will be.

        """
        CH_list = []
        for i in range(self.K):
            CH = node.ClusterHead()
            # generate a random phi and a random radius
            phi_CH = np.random.uniform(
                low=i * np.pi / self.K, high=(i + 1) * np.pi / self.K
            )
            r_CH = r_max * np.random.uniform()
            CH.x = r_CH * np.cos(phi_CH)
            CH.y = r_CH * np.sin(phi_CH)
            CH_list.append(CH)

        self.CH_list = CH_list

    def generate_and_assign_coefs(self, random_angles: bool = False) -> None:
        """
        Given positions of CHs, the AoA angles are generated and all the
        objects BS, RIS, CHs will be updated with values.

        Args:
            random_angles (bool, optional): [Option to specify the randomness
                of all AoA angles]. Defaults to False.
        """

        channel = cl.Channel(self.config)

        # Generate neccessary data
        channel.generate_Rician_coefs(
            RIS=self.RIS, node=self.BS, random_angles=random_angles
        )
        for CH in self.CH_list:
            channel.generate_Rician_coefs(
                RIS=self.RIS, node=CH, random_angles=random_angles
            )

    def sketch(self) -> None:
        """
        Sketch the positions of the network.
        """
        fig, ax = plt.subplots()
        CH_area = plt.Circle(
            (0, 0), self.config["system_model"]["CH"]["r_max"], fill=False
        )
        ax.add_patch(CH_area)
        # Use adjustable='box-forced' to make the plot area square-shaped as well.
        ax.set_aspect("equal", adjustable="datalim")
        ax.plot()  # Causes an autoscale update.
        # RIS
        ax.scatter(self.RIS.x, self.RIS.y, color="blue")
        # BS
        ax.scatter(self.BS.x, self.BS.y, color="red")
        # CHs
        for CH in self.CH_list:
            ax.scatter(CH.x, CH.y, color="black")
        plt.grid()
        plt.show()

    def generate_parameters(self) -> LargeScaleCoef:
        """
        Generate a LargeScaleCoef object containing the large scale coefs
        which can be used for further analysis.

        Returns:
            LargeScaleCoef
        """
        f = np.array(
            list(
                map(
                    lambda x: self.RIS.num_phase
                    * self.BS.beta
                    * x.beta
                    * (self.BS.K + x.K + 1)
                    / (x.K + 1)
                    / (self.BS.K + 1)
                    * self.BS.power
                    / self.noise_power,
                    self.CH_list,
                )
            )
        ).reshape((self.K, 1))

        r = np.array(
            list(
                map(
                    lambda x: self.BS.power
                    / self.noise_power
                    * self.BS.beta
                    * x.beta
                    * self.BS.K
                    * x.K
                    / (self.BS.K + 1)
                    / (x.K + 1),
                    self.CH_list,
                )
            )
        ).reshape((self.K, 1))

        # Element wise product
        angle_matrix = np.array(
            list(
                map(
                    lambda x: x.AoA_array.conj() * self.BS.AoA_array,
                    self.CH_list,
                )
            )
        )
        angle_matrix = angle_matrix.reshape((self.K, self.RIS.num_phase)).transpose()
        angle_matrix = angle_matrix.conj().transpose()

        # Cascaded angle
        cascaded_AoA = np.array(
            list(
                map(
                    lambda x: np.sin(x.AoA) - np.sin(self.BS.AoA),
                    self.CH_list,
                )
            )
        )

        return LargeScaleCoef(F=f, R=r, A=angle_matrix, cascaded_AoA=cascaded_AoA)

    def calculate_ergodic_SE_upper_bound_each_user(self) -> np.array:
        """
        Calculate the upper bound ergodic spectral efficiency of each user
        for a given BS, CHs and RIS.

        Returns:
            np.array: [Upper bound ergodic spectral efficiency of each user]
        """
        vector_phi = self.RIS.get_phi()[0]
        # get the parameters which is in a large scale coefs
        large_scale_coefs = self.generate_parameters()
        # the term containing the angle and phi
        var_term = (
            np.linalg.norm(np.mat(large_scale_coefs.A) * np.mat(vector_phi), axis=1)
            ** 2
        ).reshape((self.K, 1))

        return np.log2(1 + large_scale_coefs.F + large_scale_coefs.R * var_term)

    def calculate_ergodic_SE_upper_bound_analytic(self) -> float:
        """
        Calculate the upper bound for ergodic SE for a given BS, CHs and
        RIS by using the analytic method.

        Returns:
            float: [The upper bound of ergodic SE]
        """
        vector_phi = self.RIS.get_phi()[0]
        # get the parameters
        large_scale_coefs = self.generate_parameters()
        # the term containing the angle and phi
        var_term = (
            np.linalg.norm(np.mat(large_scale_coefs.A) * np.mat(vector_phi), axis=1)
            ** 2
        ).reshape((self.K, 1))

        return np.sum(np.log2(1 + large_scale_coefs.F + large_scale_coefs.R * var_term))

    def calculate_ergodic_SE_upper_bound_simulation(
        self, n_loop: int = 1000, print_option: bool = False
    ) -> float:
        """
        Calculate the upper bound for ergodic SE for a given BS, CH and
        RIS by using the simulation method.

        Args:
            n_loop (int, optional): [Number of loops]. Defaults to 10000.
            print_option (bool): [Print time option]. Defaults to False

        Returns:
            float: [The upper bound of ergodic SE]
        """
        start = time.time()

        matrix_phi = self.RIS.get_phi()[-1]
        channel = cl.Channel(self.config)
        total_SE_bound_simulation = 0.0

        for i in range(n_loop):
            # a random channel between RIS and BS
            h0 = channel.generate_random_Rician_channel(self.RIS, self.BS)
            # a random channel between RIS and CH
            hk = [
                channel.generate_random_Rician_channel(self.RIS, CH)
                for CH in self.CH_list
            ]

            total_SE_bound_simulation += np.array(
                [
                    self.BS.power
                    / self.noise_power
                    * (
                        np.linalg.norm(
                            np.mat(hk[i]).getH() * np.mat(matrix_phi) * np.mat(h0)
                        )
                        ** 2
                    )
                    for i in range(self.K)
                ]
            )
        end = time.time()
        if print_option:
            print("Run time -- ", end - start)

        return np.sum(np.log2(1 + total_SE_bound_simulation / n_loop))

    def calculate_ergodic_SE_upper_bound_simulation_using_vectorization(
        self, n_loop: int = 1000, print_option: bool = False
    ) -> float:
        """
        Calculate the upper bound for ergodic SE for a given BS, CH and
        RIS by using the simulation method with vectorization.

        Args:
            n_loop (int, optional): [Number of loops]. Defaults to 1000.
            print_option (bool): [Print options]. Defaults to False

        Returns:
            float: [The upper bound of ergodic SE]
        """
        start = time.time()
        vector_phi = self.RIS.get_phi()[0]
        channel = cl.Channel(self.config)
        total_SE_bound_simulation = 0.0

        for i in range(n_loop):
            # a random channel between RIS and BS
            h0 = channel.generate_random_Rician_channel(self.RIS, self.BS)
            # a random channel between RIS and CHs
            hk = np.array(
                [
                    channel.generate_random_Rician_channel(self.RIS, CH)
                    for CH in self.CH_list
                ]
            )
            qk = hk.conj() * h0
            qk = qk.reshape((self.K, self.RIS.num_phase)).transpose()
            # a random channel between BS and CHs

            total_SE_bound_simulation += (
                self.BS.power
                / self.noise_power
                * np.linalg.norm(np.mat(qk).getH() * np.mat(vector_phi), axis=1) ** 2
            )
        end = time.time()
        if print_option:
            print("Run time -- ", end - start)

        return np.sum(np.log2(1 + total_SE_bound_simulation / n_loop))

    def calculate_ergodic_SE(
        self, n_loop: int = 1000, print_option: bool = False
    ) -> float:
        """
        Calculate the ergordic SE with a given BS, CH and RIS. This is
        done by using the Monte Carlo simulation.

        Args:
            n_loop (int, optional): [Number of loops]. Defaults to 1000.
            print_option (bool): [Print options]. Defaults to False

        Returns:
            float: [The ergodic SE]
        """
        start = time.time()
        channel = cl.Channel(self.config)
        total_ergodic_SE = 0.0
        matrix_phi = self.RIS.get_phi()[-1]

        for i in range(n_loop):
            # generate a random complex channel between RIS and BS
            h0 = channel.generate_random_Rician_channel(RIS=self.RIS, node=self.BS)
            # generate a list of random complex channel between RIS and CHs
            hk = [
                channel.generate_random_Rician_channel(RIS=self.RIS, node=CH)
                for CH in self.CH_list
            ]

            total_ergodic_SE += sum(
                np.log2(
                    1
                    + self.BS.power
                    / self.noise_power
                    * np.linalg.norm(
                        np.mat(hk[i]).getH() * np.mat(matrix_phi) * np.mat(h0)
                    )
                    ** 2
                )
                for i in range(self.K)
            )
        end = time.time()
        if print_option:
            print("Run time -- ", end - start)

        return total_ergodic_SE / n_loop

    def calculate_ergodic_SE_using_vectorization(
        self, n_loop: int = 1000, print_option: bool = False
    ) -> float:
        """
        Calculate the ergordic SE with a given BS, CH and RIS. This is
        done by using the Monte Carlo simulation with vectorization.

        Args:
            n_loop (int, optional): [Number of loops]. Defaults to 1000.
            print_option (bool): [Print options]. Defaults to False

        Returns:
            float: [The ergodic SE]
        """
        start = time.time()
        channel = cl.Channel(self.config)
        total_ergodic_SE = 0.0
        vector_phi = self.RIS.get_phi()[0]

        for i in range(n_loop):
            # generate a random complex channel between RIS and BS
            h0 = channel.generate_random_Rician_channel(RIS=self.RIS, node=self.BS)
            # generate a list of random complex channel between RIS and CHs
            hk = np.array(
                [
                    channel.generate_random_Rician_channel(RIS=self.RIS, node=CH)
                    for CH in self.CH_list
                ]
            )
            qk = hk.conj() * h0
            qk = qk.reshape((self.K, self.RIS.num_phase)).transpose()

            total_ergodic_SE += np.sum(
                np.log2(
                    1
                    + self.BS.power
                    / self.noise_power
                    * np.linalg.norm(np.mat(qk).getH() * np.mat(vector_phi), axis=1)
                    ** 2
                )
            )

        end = time.time()
        if print_option:
            print("Run time -- ", end - start)

        return total_ergodic_SE / n_loop

    def calculate_outage_each_link_analytic(self, R_th: np.float) -> np.float:
        """
        Calculate the outage probability of transmissions for all CHs.
        P_out = Pr(SE_k < R_th)

        Args:
            R_th (np.float): [SE threshold]

        Returns:
            np.float: [Outage probability]
        """
        ls_coef = self.generate_parameters()
        F = ls_coef.F
        R = ls_coef.R
        A = ls_coef.A

        vector_phi = self.RIS.get_phi()[0]

        return np.array(
            [
                1
                - marcum_q(
                    1,
                    np.sqrt(2 * R[i] / F[i]) * np.linalg.norm(np.matmul(A[i], vector_phi)),
                    np.sqrt((2 ** (R_th + 1) - 2) / F[i]),
                )
                for i in range(self.K)
            ]
        )

    def calculate_outage_each_link_simulation(
        self, R_th: np.float, n_loop: int = 1000
    ) -> np.array:
        """
        Calculate the outage probability for the transmission to all CHs using
        simulation.

        Args:
            R_th (np.float): [SE threshold]
            n_loop (int, optional): [Number of loops]. Defaults to 1000.

        Returns:
            np.float: [Outage probability for the link]
        """
        vector_phi = self.RIS.get_phi()[0]
        channel = cl.Channel(self.config)

        def generate_qk():
            h0 = channel.generate_random_Rician_channel(RIS=self.RIS, node=self.BS)
            hk = np.array(
                [
                    channel.generate_random_Rician_channel(RIS=self.RIS, node=CH)
                    for CH in self.CH_list
                ]
            )
            qk = hk.conj() * h0
            qk = qk.reshape((self.K, self.RIS.num_phase)).transpose()
            return qk.transpose().conj()

        one_batch_qk = np.array([generate_qk() for i in range(n_loop)])
        SE = (
            self.BS.power
            / self.noise_power
            * np.abs(np.matmul(one_batch_qk, vector_phi)) ** 2
        )

        return np.sum(np.log2(1 + SE) < R_th, axis=0) / n_loop


    def calculate_rate(self) -> np.array:
        """
        Calculate the instantaneous rate of each user.
        """
        vector_phi = self.RIS.get_phi()[0]
        channel = cl.Channel(self.config)

        def generate_qk():
            h0 = channel.generate_random_Rician_channel(RIS=self.RIS, node=self.BS)
            hk = np.array(
                [
                    channel.generate_random_Rician_channel(RIS=self.RIS, node=CH)
                    for CH in self.CH_list
                ]
            )
            qk = hk.conj() * h0
            qk = qk.reshape((self.K, self.RIS.num_phase)).transpose()
            return qk.transpose().conj()

        qk = generate_qk()

        SE = (
            self.BS.power
            / self.noise_power
            * np.abs(np.matmul(qk, vector_phi)) ** 2
        )

        return np.log2(1 + SE)

    def calculate_outage_all_system_analytic(self, m: int, y_th: np.float) -> float:
        """
        Caculate the outage probability for the entire broadcast system using
        analytic method.

        Args:
            m (int): [The system is considered to be in outage if there are
                 at least m transmissions to CHs are in outage]
            y_th (np.float): [SNR threshold]

        Returns:
            float: [outage probability]
        """

        assert m <= self.K

        def findsubsets(s, n) -> List:
            """
            Return a list containing selections of n items from s
            """
            return list(itertools.combinations(s, n))

        u_set = range(self.K)
        outage_analytic = self.calculate_outage_each_link_analytic(y_th=y_th)

        subset = []
        # Get all the subset which contains more than m items from CHs
        for i in np.arange(m, self.K + 1, 1):
            subset += findsubsets(u_set, i)

        return np.sum(
            [
                np.prod(
                    list(
                        map(
                            lambda x: outage_analytic[x]
                            if x in subset_hn
                            else 1 - outage_analytic[x],
                            u_set,
                        )
                    )
                )
                for subset_hn in subset
            ]
        )
