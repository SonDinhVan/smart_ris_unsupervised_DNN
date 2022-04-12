"""
This class provides the set-up for nodes (i.e. BS, CH, RIS) in the network.
"""
import numpy as np
from typing import List
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Node:
    """
    A data class to contain nodes in the network system.

    Args:
        x (float, optional): x-coordinator position of node. Default = 0.
        y (float, optional): y-coordinator position of node. Default = 0.
    """

    x: float = 0.0
    y: float = 0.0

    def get_distance(self, node: "Node") -> float:
        """
        Get distance from itself to another node

        Args:
            node_2 (node): A node in the network

        Returns:
            float: the distance (in meters)
        """
        return np.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)

    def get_cos_angle(self, node: "Node") -> float:
        """
        Get the cosin value of the angle of the signal between nodes

        Args:
            node_2 (node): A node in the network

        Returns:
            float: [The cosin of angle of the signal respect to the Horizontal line]
        """
        return np.abs(self.x - node.x) / self.get_distance(node)

    def get_angle(self, node: "Node") -> float:
        """
        Get the angle of the direct signal between two nodes

        Args:
            node (Node): [A node in the network]

        Returns:
            float: [The angle of the signal respect to the Horizontal line]
        """
        return np.arccos(np.abs(self.x - node.x) / self.get_distance(node))


@dataclass
class ClusterHead(Node):
    """
    ClusterHead data class

    Args:
        beta (float, optional): [the large scale coef associated to the link
            from CH to RIS]. Defaults to None.
        K (float, optional): [The K parameter for the link from CH to RIS].
            Defaults to None.
        AoA_array(np.array, optional): [The response of Angle of Arriving signal
            from the CH to RIS]. Defaults to None.
        AoA(np.float, optional): [The Angle of Arriving signal from the
            RIS to CH]. Defaults to None.
        h_bar(np.array, optional): [The mean of rician coef from CH to RIS].
            Defaults to None.
    """

    beta: float = None
    K: float = None
    AoA_array: np.array = None
    AoA: np.float = None
    h_bar: np.array = None

    def generate_a_random_position(self, x0: float = None, y0: float = None) -> None:
        """
        Generate a random position of CH in a square space with the x-location
        and y-location are uniformly distributed.

        Args:
            x0 (float, optional): [Range of x-coordinator]. Defaults to None.
            y0 (float, optional): [Range of y-coordinator]. Defaults to None.
        """
        self.x = x0 * np.random.uniform()
        self.y = y0 * np.random.uniform()


@dataclass
class BaseStation(Node):
    """
    Base Station data class

    Args:
        power (float, optional): [Transmit power in Watt].
            Defaults to None.
        beta (float, optional): [The large scale coef associated to the link
            from BS to RIS]. Defaults to None.
        K (float, optional): [The K parameter for the link from BS to RIS].
            Defaults to None.
        AoA_array(np.array, optional): [The array response of Angle of Arriving
            signal from the BS to RIS]. Defaults to None.
        AoA(np.float, optional): [The Angle of Arriving signal from the BS to
            RIS]. Defaults to None.
        h_bar(np.array, optional): [The mean of rician coef from BS to RIS].
            Defaults to None.
    """

    power: float = None
    beta: float = None
    K: float = None
    AoA_array: np.array = None
    AoA: np.float = None
    h_bar: np.array = None


@dataclass
class ReIntelSurface(Node):
    """
    Reconfigurable Intelligent Surface class

    Args:
        num_phase (int, optional): [Number of phase elements. Defaults = None]
        phase (np.array, optional): [Vector of phase elements
            Phi = [phi_1, phi_2, ..., phi_N]^T. Defaults to None]
    """

    num_phase: int = None
    phase: np.array = None

    def get_random_phase(self) -> None:
        """
        Use a random phase with each phi_i is uniformly distributed in
        [0, 2 * pi]
        """
        self.phase = np.random.uniform(
            low=0.0, high=2 * np.pi, size=(self.num_phase, 1)
        )

    def get_phi(self) -> List:
        """
        Get the vector phi and diagonal matrix phi from the phase of RIS
        Vector phi = [np.exp(1j * phi_0), ...]^T

        Returns:
            List: [vector phi, diagonal matrix phi]
        """
        vector_phi = np.exp(1j * self.phase)
        return vector_phi, np.diagflat(vector_phi)

    def get_quantized_phase(self, phase: np.array, Q: int) -> np.array:
        """
        Get the quantized phase by finding the closest quantized value
        to the continuous phase.

        Args:
            phase (np.array): [The phase values will be converted into
                quantized space]
            Q (int): [Number of bits]
        
        Returns:
            Quantized phase values
        """
        quantized_phase = 2 * np.pi / 2 ** Q * np.arange(0, 2 ** Q, 1)

        def find_closest(c):
            return quantized_phase[np.argmin(abs(c - quantized_phase))]

        return np.array(list(map(lambda x: find_closest(x), phase))).reshape(
            (self.num_phase, 1)
        )
