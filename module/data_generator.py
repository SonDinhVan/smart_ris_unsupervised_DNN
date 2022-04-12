"""
This class is to generate training data for a given network setup.
"""
import numpy as np
from module import system
from typing import List


class DataGenerator:
    """
    This class generates training data.
    """

    def __init__(self, network: system.Network) -> None:
        """
        Initialization. This class requires a set up of network topology
        to operate (i.e. num_phase, number of CHs).

        Args:
            network_topology (sty.NetworkTopology): [The input network_topology]
        """
        self.network = network

    def generate_one_sample_data(
        self,
        r_max: np.float = 50.0,
        power_change: bool = True,
        cascaded_data: bool = True,
        random_angles: bool = True,
    ) -> List:
        """
        Generates one training data sample based on one realization of the
        given network topology.

        Args:
            r_max (np.float): [The maximum radius of the area]. Defaults to
                50 m.
            power_change (bool, optional): [Option to consider power change to
                generate training data]. Defaults to True.
            cascaded_data (bool, optional): [Option to consider the data
                generated is A or cascaded_AoA. Using cascaded_AoA will save a
                significant memory compared to using A.] Defaults to True.
            random_angles (bool, optional): [Option to consider random angles
                to generate training data]. Defaults to True.

        Returns:
            np.array: [F, R, cascaded_AoA] if cascaded_data = True
            np.array: [F, R, A] if cascaded_data = False
        """
        if power_change:
            P_min = self.network.config["system_model"]["BS"]["P_min"]
            P_max = self.network.config["system_model"]["BS"]["P_max"]
            # Generate random power for the base station in a range using
            # uniform distribution
            power = np.random.uniform(P_min, P_max)
            # Assign the value in Watt into the power of BS
            self.network.BS.power = 10.0 ** (power / 10) / 1000

        self.network.generate_one_CHs_realization(r_max=r_max)
        self.network.generate_and_assign_coefs(random_angles=random_angles)
        large_scale_coef = self.network.generate_parameters()

        if cascaded_data:
            # only save the cascaded_AoA, will save a significant memory
            return [
                large_scale_coef.F,
                large_scale_coef.R,
                large_scale_coef.cascaded_AoA,
            ]
        else:
            # save the entire matrix A for training
            return [
                large_scale_coef.F,
                large_scale_coef.R,
                large_scale_coef.A,
            ]

    def generate_training_data(
        self,
        r_max: np.float = 50.0,
        num_samples: int = 1000,
        cascaded_data: bool = True,
        random_angles: bool = True,
    ) -> List:
        """
        Generates num_samples data samples.

        Args:
            r_max (np.float): [The maximum radius of the area]. Defaults to
                50 m.
            num_samples (int, optional): [Number of training samples].
                Defaults to 1000.
            cascaded_data (bool, optional): [Option to generate cascaded
                AoA or A]. Defaults to True - cascaded_AoA
            random_angles (bool, optional): [Option to specify the random
                angles or not]. Defaults to True

        Returns:
            List: [F_train, R_train, A_train]
                F_train.shape = (num_samples, K, 1)
                R_train.shape = (num_samples, K, 1)
                If cascaded_data = False
                    A_train.shape = (num_samples, K, num_phase)
                Else:
                    A_train.shape = (num_samples, K, 1)
        """

        training_data = [
            self.generate_one_sample_data(
                r_max, cascaded_data=cascaded_data, random_angles=random_angles
            )
            for i in range(num_samples)
        ]

        F_train = np.array(list(map(lambda x: x[0], training_data)))
        R_train = np.array(list(map(lambda x: x[1], training_data)))
        A_train = np.array(list(map(lambda x: x[2], training_data)))

        return F_train, R_train, A_train


def generate_A_from_cascasded_AoA(cascaded_AoA: np.array, num_phase: int) -> np.array:
    """
    Generate angle data from cascaded angle

    Args:
        cascaded_AoA (np.array): [cascaded_AoA.shape = (num_samples, K)]
        num_phase (dict): [Number of phase]

    Returns:
        np.array: [shape = (num_samples, K, num_phase)]
    """
    return np.stack(
        [np.exp(-1j * np.pi * m * cascaded_AoA) for m in range(num_phase)],
        axis=2,
    )
