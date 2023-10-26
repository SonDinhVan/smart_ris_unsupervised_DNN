- This repository contains the source code (modules and scripts) for 2 main algorithms adopted in the paper:

- S. Dinh-Van, T. M. Hoang, R. Trestian and H. X. Nguyen, "Unsupervised Deep Learning-based Reconfigurable Intelligent Surface Aided Broadcasting Communications in Industrial IoTs," in IEEE Internet of Things Journal, doi: 10.1109/JIOT.2022.3169276.

- Abstract: This article presents a general system framework that lays the foundation for reconfigurable intelligent surface (RIS)-enhanced broadcast communications in Industrial Internet of Things (IIoTs). In our system model, we consider multiple sensor clusters co-existing in a smart factory where the direct links between these clusters and a central base station (BS) are blocked completely. In this context, a RIS is utilized to reflect signals broadcast from BS toward cluster heads (CHs) which act as a representative of clusters, where BS only has access to the statistical distribution of the channel state information (CSI). An analytical upper bound of the total ergodic spectral efficiency (SE) and an approximation of outage probability are derived. Based on these analytical results, two algorithms are introduced to control the phase shifts at RIS, which are the Riemannian conjugate gradient (RCG) method and the deep neural network (DNN) method. While the RCG algorithm operates based on the conventional iterative method, and the DNN technique relies on unsupervised deep learning (DL). Our numerical results show that both algorithms achieve satisfactory performance based on only statistical CSI. In addition, compared to the RCG scheme, using DL reduces the computational latency by more than ten times with an almost identical total ergodic SE achieved. These numerical results reveal that while using the conventional RCG method may provide unsatisfactory latency, and the DNN technique shows much promise for enabling RIS in ultrareliable and low-latency communications (URLLC) in the context of IIoTs.

- Two algorithms are developed: For Unsupervised DNN, refer to module/ml_model.py; and for Riemannian Conjugate Gradient, refer to module/riemannian.py

- Do not forget to install the entire package before use via: "pip install -e. "

- To train DNN, locate "notebooks/train_DNN.ipynb"

- For a simple performance test, locate "notebooks/run_DNN_vs_Optimal_single_user.ipynb"
