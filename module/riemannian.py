"""
This class provides the solutions for the optimization using the Riemannian
conjugate gradient method. The objective is to find the mimimum value of the
following objective function
- Sum(log2(1 + F + R * |A * phi|**2))

References:
[1] https://www.researchgate.net/publication/323384783_A_Riemannian_Conjugate_Gradient_Algorithm_with_Implicit_Vector_Transport_for_Optimization_on_the_Stiefel_Manifold
[2] https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
[3] https://www.epfl.ch/labs/anchp/wp-content/uploads/2018/05/part3-1.pdf
[4] https://core.ac.uk/download/pdf/39322105.pdf "A new, globally convergent Riemannian conjugate gradient method"
[5] https://arxiv.org/pdf/2009.01451.pdf Sufficient Descent Riemannian Conjugate Gradient Methods
[6] https://arxiv.org/pdf/1706.02900.pdf An Efficient Manifold Algorithm for Constructive Interference based Constant Envelope Precoding

"""
import numpy as np
from typing import List


class Riemannian:
    def __init__(self, alpha: float, epsilon: float, max_iter: int) -> None:
        # Search step size
        self.alpha = alpha
        # The tolerance to stop the iterative process
        self.epsilon = epsilon
        # Maximum allowable number of iterations
        self.max_iter = max_iter

    def calculate_Euclidean_gradient(
        self, A: np.array, R: np.array, F: np.array, phi: np.array
    ) -> np.array:
        """
        Calculate Euclidean gradient of the objective function at phi

        Args:
            phi (np.array): [The value in which the gradient is calculated at]

        Returns:
            np.array: [The Euclidean gradient]
        """
        K = R.shape[0]
        M = A.shape[1]

        upper = (
            2
            * np.log2(np.e)
            * A.transpose().conjugate()
            * np.matmul(A, phi).reshape(K)
            * R.reshape(K)
        )
        lower = (np.square(np.abs(np.matmul(A, phi))) * R + F + 1).reshape(K)

        return -np.sum(upper / lower, axis=1).reshape((M, 1))

    def calculate_Riemannian_gradient(
        self, A: np.array, R: np.array, F: np.array, phi: np.array
    ) -> np.array:
        """
        Calculate Riemannian gradient

        Args:
            phi (np.array): [The value in which the gradient is calculated at]

        Returns:
            np.array: [The Riemannian gradient]
        """
        Eu_grad = self.calculate_Euclidean_gradient(A, R, F, phi)

        return Eu_grad - np.real(Eu_grad * phi.conjugate()) * phi

    def calculate_objective_function(
        self, A: np.array, R: np.array, F: np.array, phi: np.array
    ) -> float:
        """
        Calculate the objective function given A, R, F and phi

        Args:
            phi (np.array): []

        Returns:
            float: [description]
        """
        return -np.sum(np.log2(np.square(np.abs(np.matmul(A, phi))) * R + F + 1))

    def solve(
        self,
        A: np.array,
        R: np.array,
        F: np.array,
        print_option: bool,
        alpha_is_fixed: bool,
        initialize_option: str = "random",
    ) -> List:
        """
        Solve the optimization problem

        Args:
            print_option (bool): If True, print system performance over
                iteration number.
            alpha_is_fixed (bool): If True, fix the alpha. Else search alpha
                using Armijo condition.
            initialize_option (str): If "random", initialize random phase
                                     If "phase", initialize using phase for
                                     1 CH

        Returns:
            List: [optimal phi, optimal achieved performance]
        """

        M = A.shape[1]
        K = A.shape[0]

        num_iter = 0
        # Initiate a random phi
        angle_phi = np.random.uniform(low=0.0, high=2 * np.pi, size=(M, 1))

        if initialize_option == "phase":
            c = int(M / K)
            for i in range(K):
                angle_phi[c * i : c * (i + 1)] = -np.angle(A[i]).reshape((M, 1))[
                    c * i : c * (i + 1)
                ]
        phi = np.exp(1j * angle_phi)

        # Find search direction d at current phi
        Rie_gradient = self.calculate_Riemannian_gradient(A, R, F, phi)
        d = -Rie_gradient
        if print_option:
            print(
                "System performance : {:2f} | Norm of gradient: {:2f}".format(
                    -self.calculate_objective_function(A, R, F, phi),
                    np.linalg.norm(Rie_gradient),
                )
            )

        # Start iterations
        while np.linalg.norm(Rie_gradient) > self.epsilon:
            current_performance = self.calculate_objective_function(A, R, F, phi)
            if num_iter > self.max_iter:
                # break if the number of iteration exceeds a threshold
                print("Maximum iteration reached")
                break
            if print_option:
                if num_iter % 500 == 0:
                    print(
                        "System performance : {:2f} | Norm of gradient: {:2f}".format(
                            -self.calculate_objective_function(A, R, F, phi),
                            np.linalg.norm(Rie_gradient),
                        )
                    )

            # Calculate Rie gradient at the current phi
            Rie_gradient = self.calculate_Riemannian_gradient(A, R, F, phi)

            if alpha_is_fixed:
                alpha = self.alpha
            else:
                # Search the alpha which guarantees the Armijo rule
                beta_para = 0.5
                c1 = 0.5
                alpha = 1.0
                count = 0
                while True:
                    count += 1
                    if count > 40:
                        alpha = self.alpha
                        break
                    Armijo_condition = (
                        self.calculate_objective_function(A, R, F, phi + alpha * d)
                        - self.calculate_objective_function(A, R, F, phi)
                        - alpha
                        * c1
                        * np.abs(
                            np.matmul(
                                self.calculate_Riemannian_gradient(
                                    A, R, F, phi
                                ).transpose(),
                                np.real(d),
                            )
                        )[0][0]
                    )
                    if Armijo_condition < 0:
                        break
                    else:
                        alpha *= beta_para

            # Find the next phi
            next_phi = (phi + alpha * d) / np.abs(phi + alpha * d)
            # Calculate Rie gradient at the new phi
            next_Rie_gradient = self.calculate_Riemannian_gradient(A, R, F, next_phi)
            # Vector transport
            vector_transport = d - np.real(d * next_phi.conjugate()) * next_phi
            # Polak - Riebiere
            beta = (
                np.matmul(
                    next_Rie_gradient.transpose(),
                    (next_Rie_gradient - Rie_gradient),
                )
                / np.abs(Rie_gradient) ** 2
            )
            # Fletcher - Reeves
            # beta = np.abs(next_Rie_gradient)**2 / np.abs(Rie_gradient)**2
            beta = beta[0][0]
            # Conjugate search direction
            next_d = -next_Rie_gradient + beta * vector_transport

            # Calculate the new performance achieved
            next_performance = self.calculate_objective_function(A, R, F, next_phi)
            if np.abs(next_performance - current_performance) < self.epsilon:
                if print_option:
                    print("Break due to the convergence of performance")
                break

            # k = k + 1
            phi = next_phi
            d = next_d
            num_iter += 1

        # in the case when the while loop was skipped (this occurs in the
        # case when the initialization is too close to optimal solution)
        if num_iter == 0:
            return phi, -self.calculate_objective_function(A, R, F, phi)
        # otherwise
        if current_performance < next_performance:
            return phi, -current_performance
        else:
            return next_phi, -next_performance
