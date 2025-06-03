
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import bokeh.application.handlers
import bokeh.io
import bokeh.models
from scipy import interpolate
from systems import systems as hs

# ---------------------------
# Habituating systems
# [1] M. Smart, S.Y. Shvartsman, & M. Monnigmann, Minimal motifs for habituating systems,
#      Proc. Natl. Acad. Sci. U.S.A. 121 (41) e2409330121, https://doi.org/10.1073/pnas.2409330121 (2024).
# ---------------------------

# Connected Minimal motif (Eq. 3 with dy/dt [1])
def CMinimal(x,t,inp,a):
  s = a[-1]
  x1_rhs = a[0]*inp-a[1]*x[0]
  y1 = inp/(1+x[0]**a[2])
  x2_rhs = a[3]*y1-a[4]*x[1]
  x3_rhs = s*((y1/(1+x[1]**a[2]))-x[2])
  return np.array([x1_rhs,x2_rhs,x3_rhs])

def CMinimal_no_delay(X, t, d, a, inp):
  input = inp(t)
  x1,x2,x3 = X(t)
  x = [x1,x2,x3]
  rhs = hs.GCinimal(x,t,input,a)
  return rhs

def CMinimal_delay_negative(X, t, d, C, a, inp):
    x1, x2, x3 = X(t)
    x1d, x2d, x3d  = X(t - d)
    s = a[-1]
    delay_term = C*x2d
    x1_rhs = a[0]*inp-a[1]*x1
    y1 = inp/(1+x1**a[2])
    x2_rhs = a[3]*y1-a[4]*x2
    x3_rhs = s*((y1/(1+x2**a[2]))-x3)
    return np.array([x1_rhs,x2_rhs, x3_rhs])

# NegativeFB (Negative feedback[1])
def NegativeFB_no_delay(X, t, d, a, inp):
    input = inp(t)
    x1,x2,x3 = X(t)
    x = [x1,x2,x3]
    rhs = hs.NegativeFB(x,t,input,a)
    return rhs

def NegativeFB_delay_negative(X, t, d, C, a, inp):
    x1, x2, x3 = X(t)
    x1d, x2d, x3d = X(t - d)
    s = a[-1]
    delay_term = C*x2d
    x1_rhs_d = (a[0]*inp(t)-a[2]*x1)
    x2_rhs_d = (a[1]*x3-a[3]*(x2-delay_term))
    x3_rhs_d = (a[4]*x1-a[5]*x2*x3)
    return np.array([s*x1_rhs_d, s*x2_rhs_d, s*x3_rhs_d])
