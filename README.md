Python code for "ARSM: Augment-REINFORCE-Swap-Merge Estimator for Gradient Backpropagation Through Categorical Variables," submitted to ICML 2019.

This file include 3 files folder: rl, toy and vae.  Codes in toy and vae folder are written with Python 2.7 and Tensorflow version 1.12.0. Codes in rl folder are written with Python 3.6 and Tensorflow version 1.8.0.


1. toy folder contains three python files: ARSM_Univariate_demo.py, ARSM_Univariate.py and toy_relax_single.py to reproduce Figure 1.
-- ARSM_Univariate_demo.py contains REINFORCE, AR , ARS and ARSM. (number of category is C =1000 in ARSM_Univariate_demo.py to show good performance with large number of categories,  C = 30 used for Figure 1).
-- ARSM_Univariate.py contains REINFORCE, AR , ARS, ARSM and Gumbel-Softmax
-- toy_relax_single.py contains RELAX

2. vae has 7 files: ar_vae.py, ars_vae.py,arsm_vae.py, arsm_vae_l2.py,  gumbel_vae.py, relax_vae.py, and rf_vae.py to reproduce Figure 2.
-- rf_vae.py is REINFORCE
-- ar_vae.py is the AR estimator
-- ars_vae.py is the ARS estimator
-- arsm_vae.py is the ARSM estimator
-- gumbel_vae.py is Gumbel-Softmax
-- relax_vae.py is RELAX

3. rl folder contains two python files: arm_util.py and RL_ARSM.py to reproduce Figure 3.
-- arm_util.py contains functions used in RL_ARSM.py.
-- RL_ARSM.py contains ARS and ARSM algorithm on RL task, default task is Cart Pole.

Under review, pleaes do not distribute.
