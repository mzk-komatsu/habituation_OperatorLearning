
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import bokeh.application.handlers
import bokeh.io
import bokeh.models
from scipy import interpolate

# ---------------------------
# Habituating systems
# [1] M. Smart, S.Y. Shvartsman, & M. Monnigmann, Minimal motifs for habituating systems, 
#      Proc. Natl. Acad. Sci. U.S.A. 121 (41) e2409330121, https://doi.org/10.1073/pnas.2409330121 (2024).
# ---------------------------

#Tyson-a-L-2D (Sniffer[1])
def TysonAL2D(x,t,inp, a):
  s = a[-1]
  x1_rhs = s*(a[0]*inp-a[2]*x[0])
  x2_rhs = s*(a[1]*inp-a[3]*x[0]*x[1]; #a[1]*inp-a[3]*x[0]*x[1])
  return np.array([x1_rhs, x2_rhs])

# NegativeFB (Negative feedback[1])
def NegativeFB(x, t, inp, a):
  s = a[-1]
  x1_rhs = s*(a[0]*inp-a[2]*x[0])
  x2_rhs = s*(a[1]*x[2]-a[3]*x[1])
  x3_rhs = s*(a[4]*x[0]-a[5]*x[1]*x[2])
  return np.array([x1_rhs, x2_rhs, x3_rhs])

# Generalized Minimal motif (Eq.2 with dy/dt [1])
def GMinimal(x,t,inp,a):
  s = a[-1]
  x1_rhs = a[0]*inp-a[1]*x[0]
  x2_rhs = s*((inp/(1+x[0]**a[2]))-x[1])
  return np.array([x1_rhs,x2_rhs])

# Connected Minimal motif (Eq. 3 with dy/dt [1])
def CMinimal(x,t,inp,a):
  s = a[-1]
  x1_rhs = a[0]*inp-a[1]*x[0]
  y1 = inp/(1+x[0]**a[2])
  x2_rhs = a[3]*y1-a[4]*x[1]
  x3_rhs = s*((y1/(1+x[1]**a[2]))-x[2])
  return np.array([x1_rhs,x2_rhs,x3_rhs])


# -----------------------------------------------------
# Habituating systems given input function
# input function: interporated repetitive stimuli
# -----------------------------------------------------

def TysonAL2D_input_interporate(x,t, a0, xt, xf):
    #print("TysonAL2D_input_interporate:", x, "<==")
    interp = interpolate.interp1d(np.array(xt), np.array(xf),fill_value='extrapolate')
    input = interp(t)
    return TysonAL2D(x,t,input, a0)

def NegativeFB_input_interporate(x, t, a0, xt, xf):
    interp = interpolate.interp1d(np.array(xt), np.array(xf),fill_value='extrapolate')
    input = interp(t)
    return NegativeFB(x, t, input, a0)

def GMinimal_input_interporate(x, t, a0, xt, xf):
    interp = interpolate.interp1d(np.array(xt), np.array(xf),fill_value='extrapolate')
    input = interp(t)
    return GMinimal(x, t, input, a0)
  
def CMinimal_input_interporate(x, t, a0, xt, xf):
    interp = interpolate.interp1d(np.array(xt), np.array(xf),fill_value='extrapolate')
    input = interp(t)
    return CMinimal(x, t, input, a0)


def get_rhs(sys):
  if sys == "TysonAL2D":
    rhs_inp = TysonAL2D_input_interporate
  elif sys == "NegativeFB":
    rhs_inp = NegativeFB_input_interporate
  elif sys == "GMinimal":
    rhs_inp = GMinimal_input_interporate
  elif sys == "CMinimal":
    rhs_inp = CMinimal_input_interporate
  return rhs_inp
  


