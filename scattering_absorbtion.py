import numpy as np

# RZ11
# total = 3.96, T = 0.27, T.directly = 0.18, R = 0.5
T = 1  # transmittance
R = 1  # reflectance
A = 1  # absorbance
TF = 1 - A  # transflectance
length = 1  # in m
# L_A = -length / np.log(TF)  # absorption length
# L = -length / np.log(T)  # attenuation length
# L_S = L * L_A / (L_A - L)  # scattering length
# Z7
# total = 8.41, T = 5.54, T.directly = 3.62-0.03, R = 1.61-0.03
total = 8.41
T = 5.54 / total
T_dir = (3.62 - 0.03) / total
R = (1.61 - 0.03) / total

A = 1 - T - R
TF = 1 - A
length = 1  # in m
# L_A = -length / np.log(TF)  # absorption length
# L = -length / np.log(T)  # attenuation length
# L_S = L * L_A / (L_A - L)  # scattering length
print(T+R)

