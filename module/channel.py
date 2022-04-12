"""
This class provides useful functions for generating the channel coefficients
, i.e. Path Loss, K-parameters, small-scale, shadowing.
...
"""
import numpy as np
from module import node
from typing import Dict


class Channel:
    """
    Channel model class
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialization

        Args:
            config (Dict): Dictionary of config imported from yaml
        """
        # Convert to Hz
        self.f = config["system_model"]["f"] * 10.0 ** 9
        self.c = config["constant"]["c"]
        self.standard_shadow_db = config["channel"]["standard_shadow_db"]

    def generate_plos(self, d: float) -> float:
        """
        Generate the line of sight probability (p_los)

        Args:
            d (float): distance of the communication link

        Returns:
            float: line of sight probability
        """
        if d <= 1.0:
            return 0.95
        elif d < 9.8:
            return np.exp((1 - d) / 4.9)
        else:
            return 0.17

    def generate_K(self, d: float) -> float:
        """
        Generate the K parameter for the communication link

        Args:
            d (float): distance of the communication link

        Returns:
            float: the parameter K
        """
        p_los = self.generate_plos(d)
        return p_los / (1 - p_los)

    def generate_beta(self, d: float) -> float:
        """
        Generate the beta coef of the link
        PL = 71.84 + 21.6 * log10(d/15)
        Paper link: E. Tanghe et. al
        “The industrial indoor channel: large-scale and temporal fading at 900, 2400, and 5200 MHz,”
        IEEE Transactions on Wireless Communications, vol. 7, no. 7, pp. 2740–2751, 2008.
        shadow_val = 10 ** (z/10)
        where z = standard_shadow_db * normal_random

        Args:
            d (float): distance of the communication link

        Returns:
            float: the beta coef
        """
        # path-loss of the link
        PL_db = 71.84 + 21.6 * np.log10(d / 15)
        PL = 10.0 ** (-PL_db / 10)

        # shadowing of the link
        z = self.standard_shadow_db * np.random.normal(0, 1)
        shadow_val = 10.0 ** (z / 10)

        return shadow_val * PL

    def generate_Rician_coefs(
        self,
        RIS: node.ReIntelSurface,
        node: node.Node,
        random_angles: bool = True,
    ) -> None:
        """
        This will parse the value of the K, beta, AoA_array and h_bar into the
        attributes of CH or BS.

        Args:
            RIS (system.RIS): [A RIS]
            node (system.node): [A node - only can be CH or BS]
            random_angles (bool): [If True - generate all random AoA
                False - get the direct angles]. Defaults = True
        """
        # distance from RIS to node
        dis = RIS.get_distance(node=node)
        # Parameter beta
        node.beta = self.generate_beta(dis)
        # Parameter K
        node.K = self.generate_K(dis)
        # Get the angle AoA by random or direct angles
        if random_angles:
            # generate random angles
            node.AoA = np.random.uniform(0, 2 * np.pi)
        else:
            node.AoA = RIS.get_angle(node)

        # get the AoA array from the angle, this is column vector
        node.AoA_array = np.array(
            [np.exp(-1j * np.pi * m * np.sin(node.AoA)) for m in range(RIS.num_phase)]
        ).reshape((RIS.num_phase, 1))

        # Mean of the channel h_bar
        node.h_bar = np.sqrt(node.beta * node.K / (node.K + 1)) * node.AoA_array

    def generate_random_Rician_channel(
        self, RIS: node.ReIntelSurface, node: node.Node
    ) -> np.array:
        """
        Generate an instantaneous Rician channel. It is not recommended to run
        this before runing the function generate_Rician_coefs.

        Args:
            RIS (system.RIS): [a RIS]
            node (system.node): [only can be a CH or a BS]

        Returns:
            np.array: [a vector of complex Rician channel]
        """

        z = (
            np.random.normal(size=[RIS.num_phase, 1]) / np.sqrt(2)
            + np.random.normal(size=[RIS.num_phase, 1]) / np.sqrt(2) * 1j
        )
        return node.h_bar + np.sqrt(node.beta / (node.K + 1)) * z
