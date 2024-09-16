################################################################
# Case 2: RLC                                                  #
################################################################
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

def pulse(t,T): # square pulse, T-period
  if len(np.shape(t)) == 0: y = 0.5 if (t % T) <= T/2 else -0.5  # scalar
  else: # vector
    y = np.zeros(shape=(len(t),))
    for i in range(len(t)): y[i] = 0.5 if (t[i] % T) <= T/2 else -0.5
  return y
def ramp(t,T): # ramp signal, T-period
  if len(np.shape(t)) == 0: y = (t % T)/T # scalar
  else: # vector, only for figure
    y = np.zeros(shape=(len(t),))
    for i in range(len(t)): y[i] = (t[i] % T)/T
  return y
def staircase(t):
  if len(np.shape(t)) == 0: # scalar
    if (t>=0) & (t<20.0): y = 1.0
    elif (t>=20.0) & (t<25.0): y = 2.0
    elif (t>=25.0) & (t<30.0): y = 3.0
    elif t>=30.0: y = 1.0
  else: # vector, only for figure
    y = np.zeros(shape=(len(t),))
    for i in range(len(t)):
      if (t[i]>=0) & (t[i]<20.0): y[i] = 1.0
      elif (t[i]>=20.0) & (t[i]<25.0): y[i] = 2.0
      elif (t[i]>=25.0) & (t[i]<30.0): y[i] = 3.0
      elif t[i]>=30.0: y[i] = 1.0
  return y

# Ground truth
# a: alpha1, b:alpha1', c: alpha2, d: alpha2', di: di^2, qi = qi
a0, b0, c0, d0 = 0.1, 0.1, 0.1, 0.1
d10, d20, q10, q20 = 1.0, 1.0, 1.0, 1.0
y0 = np.array([0.0,0.0,0.0,0.0]) # initial condition/state
t_span = [0,50]
t_eval = np.arange(0.0,50.0,0.1)
input_signal = "sin"
def func_np(t,y,a,b,c,d,d1,d2,q1,q2,input_signal): # current time, current solution, params
  if input_signal == "sin": u = np.sin(t)
  elif input_signal == "step": u = 1.0
  elif input_signal == "ramp": u = ramp(t,5.0)
  elif input_signal == "pulse": u = pulse(t,5.0)
  elif input_signal == "staircase": u = staircase(t)
  return (np.array([q1+q2,q1*(c+d+b)+q2*(a+b+d),q1*(c*d+d2+b*(c+d))+q2*(a*b+d1+d*(a+b)),q1*b*(c*d+d2)+q2*d*(a*b+d1)])*u +
          np.matmul(y,np.array([[-(a+b+c+d),-((a+b)*(c+d)+a*b+d1+c*d+d2),-((a+b)*(c*d+d2)+(c+d)*(a*b+d1)),-(a*b+d1)*(c*d+d2)],
           [1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0]])))
sol = solve_ivp(func_np, t_span=t_span, y0=y0,t_eval=t_eval,args=(a0,b0,c0,d0,d10,d20,q10,q20,input_signal))
y_GT = sol.y[0]
N_train = int(len(t_eval)/5)
t_train = t_eval[0:N_train]
y_train = y_GT[0:N_train]

# params for RK4
A = tf.constant([0.0, 2/9, 1/3, 3/4, 1.0, 5/6]) #A1, A2, A3, A4, A5, A6
Bx1 = tf.constant([2/9, 1/12, 69/128, -17/12, 65/432]) #B21, B31, B41, B51, B61 =
Bx2 = tf.constant([1/4, -243/128, 27/4, -5/16]) #B32, B42, B52, B62 =
Bx3 = tf.constant([135/64, -27/5, 13/16]) #B43, B53, B63 =
Bx4 = tf.constant([16/15, 4/27]) #B54, B64 =
Bx5 = tf.constant(5/144) # B65 =
CH1, CH2, CH3, CH4, CH5, CH6 = 47/450, 0, 12/25, 32/225, 1/30, 6/25

def func(t,y,a,b,c,d,d1,d2,q1,q2,input_signal): # current time, current solution, (a,b,c): parameters
  Be = tf.concat([[[q1+q2]],[[q1*(c+d+b)+q2*(a+b+d)]],[[q1*(c*d+d2+b*(c+d))+q2*(a*b+d1+d*(a+b))]],[[q1*b*(c*d+d2)+q2*d*(a*b+d1)]]],1)
  Ae = (tf.concat([[[-(a+b+c+d),-((a+b)*(c+d)+a*b+d1+c*d+d2),-((a+b)*(c*d+d2)+(c+d)*(a*b+d1)),-(a*b+d1)*(c*d+d2)]],
   [[1.0,0.0,0.0,0.0]],[[0.0,1.0,0.0,0.0]],[[0.0,0.0,1.0,0.0]]],0))
  if input_signal == "sin": u = tf.math.sin(t)
  elif input_signal == "step": u = tf.constant(1.0)
  elif input_signal == "ramp": u = (t % 5.0)/5.0 # T = 5
  elif input_signal == "pulse": u = tf.constant(0.5) if (t % 5.0) <= 5.0/2 else tf.constant(-0.5) # T = 5
  elif input_signal == "staircase":
    if (t>=0) & (t<20.0): u = tf.constant(1.0)
    elif (t>=20.0) & (t<25.0): u = tf.constant(2.0)
    elif (t>=25.0) & (t<30.0): u = tf.constant(3.0)
    elif t>=30.0: u = u = tf.constant(1.0)
  return Be*u + tf.linalg.matmul(y,Ae) # y must be 1x2
def my_RK45(t,y,h,a,b,c,d,d1,d2,q1,q2,input_signal): # one step
  k1 = h*func(t+A[0]*h,y,a,b,c,d,d1,d2,q1,q2,input_signal)
  k2 = h*func(t+A[1]*h,y+Bx1[0]*k1,a,b,c,d,d1,d2,q1,q2,input_signal)
  k3 = h*func(t+A[2]*h,y+Bx1[1]*k1+Bx2[0]*k2,a,b,c,d,d1,d2,q1,q2,input_signal)
  k4 = h*func(t+A[3]*h,y+Bx1[2]*k1+Bx2[1]*k2+Bx3[0]*k3,a,b,c,d,d1,d2,q1,q2,input_signal)
  k5 = h*func(t+A[4]*h,y+Bx1[3]*k1+Bx2[2]*k2+Bx3[1]*k3+Bx4[0]*k4,a,b,c,d,d1,d2,q1,q2,input_signal)
  k6 = h*func(t+A[5]*h,y+Bx1[4]*k1+Bx2[3]*k2+Bx3[2]*k3+Bx4[1]*k4+Bx5*k5,a,b,c,d,d1,d2,q1,q2,input_signal)
  return y+CH1*k1+CH2*k2+CH3*k3+CH4*k4+CH5*k5+CH6*k6
def my_RK45_time(t_eval,y_init,a,b,c,d,d1,d2,q1,q2,input_signal):
  yt = y_init
  N = len(t_eval)
  y = yt
  for i in range(N-1):
    yt = my_RK45(t_eval[i],yt,t_eval[i+1]-t_eval[i],a,b,c,d,d1,d2,q1,q2,input_signal)
    y = tf.concat([y,yt],0)
  return y  # Nx2

t_eval_ts = tf.convert_to_tensor(t_eval, dtype=tf.float32)
t_train_ts = tf.convert_to_tensor(t_train, dtype=tf.float32)
y_train_ts = tf.convert_to_tensor(y_train,dtype=tf.float32)
params = params_save = tf.Variable([0.2,0.2,0.2,0.2,1.1,0.9,1.1,0.9,0.1,0.1,0.1,0.1]) # a,b,c,d,... y_init_1,2
# load pretrained params
params = params_save = tf.Variable([0.08877416, 0.08434954, 0.11687686, 0.09408644, 1.0624727,  0.93051976,
 1.0944526,  0.89395404, 0.07403027, 0.03091692, 0.06423426, 0.0804489]) # a,b,c,d,... y_init_1,2

# TRAINING
alpha = 0.01 # learning rate
iter = 0
iter_max = 2
eps = 1e-6
params_tmp = params
while iter < iter_max:
  iter = iter + 1
  with tf.GradientTape() as g:
    g.watch(params) 
    y = (my_RK45_time(t_eval=t_train_ts,y_init=[params[8:]],a=params[0],b=params[1],c=params[2],d=params[3],
                      d1=params[4],d2=params[5],q1=params[6],q2=params[7],input_signal=input_signal))
    L = tf.reduce_mean((y[:,0]-y_train_ts)**2)
  dL_dp = g.gradient(L, params)
  params = params - alpha*dL_dp
  if abs(params[0]-params_tmp[0])<eps and abs(params[1]-params_tmp[1])<eps and abs(params[2]-params_tmp[2])<eps and abs(params[3]-params_tmp[3])<eps:
    break
  params_tmp = params
  print(*['Iteration',iter,':','(alpha,beta,gamma,delta,d1,d2,q1,q2) = ',params[0:8].numpy(),'; w_init = ',params[8:].numpy()])

print(*['Initial params: (alpha,beta,gamma,delta,d1,d2,q1,q2) = ',params_save[0:8].numpy(),'; w_init = ',params_save[8:].numpy()])
print(*['Ground truth params: (alpha,beta,gamma,delta,d1,d2,q1,q2) = (',a0,b0,c0,d0,d10,d20,q10,q20,');','w0 = ',y0])

# TESTING and RESULT
def show_results_train_test(input_signal="sin",N_plot=5000): # N_plot > N_train
  sol = solve_ivp(func_np, t_span=t_span, y0=y0,t_eval=t_eval,args=(a0,b0,c0,d0,d10,d20,q10,q20,input_signal)) # redundance
  y_GT = sol.y[0]
  y = (my_RK45_time(t_eval=t_eval_ts,y_init=[params[8:]],a=params[0],b=params[1],c=params[2],d=params[3],
                      d1=params[4],d2=params[5],q1=params[6],q2=params[7],input_signal=input_signal))
  fig, axs = plt.subplots(1,2)
  axs[0].plot(t_eval[0:N_plot],np.sin(t_eval)[0:N_plot])
  axs[0].set_title('Input signal')
  axs[0].set_xlabel('Time (s)')
  axs[1].plot(t_eval[0:N_plot],y_GT[0:N_plot], color = 'r')
  axs[1].plot(t_train_ts,y[:,0][0:N_train], color = 'g')
  axs[1].plot(t_eval_ts[N_train:N_plot],y[:,0][N_train:N_plot], color = 'b')
  plt.legend(['ground truth', 'predicted-train','predicted-test'], loc='upper right')
  axs[1].set_title('Output')
  axs[1].set_xlabel('Time (s)')

def show_results_test(input_signal="step",N_plot=5000):
  sol = solve_ivp(func_np, t_span=t_span, y0=y0,t_eval=t_eval,args=(a0,b0,c0,d0,d10,d20,q10,q20,input_signal))
  y_GT = sol.y[0]
  y = (my_RK45_time(t_eval=t_eval_ts,y_init=[params[8:]],a=params[0],b=params[1],c=params[2],d=params[3],
                      d1=params[4],d2=params[5],q1=params[6],q2=params[7],input_signal=input_signal))
  fig, axs = plt.subplots(1,2)
  if input_signal == "sin":
    axs[0].plot(t_eval[0:N_plot],np.sin(t_eval)[0:N_plot])
  elif input_signal == "step":
    axs[0].plot(t_eval[0:N_plot],1.0*np.ones(shape=(len(t_eval,)))[0:N_plot])
  elif input_signal == "ramp":
    axs[0].plot(t_eval[0:N_plot],ramp(t_eval,5.0)[0:N_plot])
  elif input_signal == "pulse":
    axs[0].plot(t_eval[0:N_plot],pulse(t_eval,5.0)[0:N_plot])
  elif input_signal == "staircase":
    axs[0].plot(t_eval[0:N_plot],staircase(t_eval)[0:N_plot])
  axs[0].set_title('Input signal')
  axs[0].set_xlabel('Time (s)')
  axs[1].plot(t_eval[0:N_plot],y_GT[0:N_plot], color = 'r')
  axs[1].plot(t_eval_ts[0:N_plot],y[:,0][0:N_plot], color = 'b')
  plt.legend(['ground truth', 'predicted-test'], loc='upper right')
  axs[1].set_title('Output')
  axs[1].set_xlabel('Time (s)')

show_results_train_test(input_signal,len(t_eval)) # default: use signal has been used for training
show_results_test("step",len(t_eval))
show_results_test("ramp",len(t_eval))
show_results_test("pulse",len(t_eval))
show_results_test("staircase",len(t_eval))
