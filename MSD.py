################################################################
# Case 1: mass spring damper                                   #
################################################################
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

# Ground truth
r10, q10 = 0.05, 10.0 # alpha1 = r1q1
d10 = np.sqrt(0.4*q10)   
y0 = np.array([0.0,0.0]) # initial condition
t_span = [0,50]
t_eval = np.arange(0.0,50.0,0.1) #0.1
input_signal = "sin"
def func_np(t,y,r1,q1,d1,input_signal): # current time, current solution
  if input_signal == "sin": u = np.sin(t)
  elif input_signal == "step": u = 1.0
  elif input_signal == "ramp": u = ramp(t,5.0)
  elif input_signal == "pulse": u = pulse(t,5.0)
  return np.array([q1,0])*u + np.matmul(y,np.array([[-r1*q1,-d1*d1],[1.0,0.0]]))
sol = solve_ivp(func_np, t_span=t_span, y0=y0,t_eval=t_eval,args=(r10,q10,d10,input_signal))
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

def func(t,y,r1,q1,d1,input_signal): # current time, current solution, (x,x,x): parameters
  Be = tf.concat([[[q1]],[[0]]],1)
  Ae = tf.concat([[[-r1*q1,-d1*d1]],[[1.0,0.0]]],0)
  if input_signal == "sin": u = tf.math.sin(t)
  elif input_signal == "step": u = tf.constant(1.0)
  elif input_signal == "ramp": u = (t % 5.0)/5.0 # T = 5
  elif input_signal == "pulse": u = tf.constant(0.5) if (t % 5.0) <= 5.0/2 else tf.constant(-0.5) # T = 5
  return Be*u + tf.linalg.matmul(y,Ae) # y must be 1x2
def my_RK45(t,y,h,r1,q1,d1,input_signal): # one step
  k1 = h*func(t+A[0]*h,y,r1,q1,d1,input_signal)
  k2 = h*func(t+A[1]*h,y+Bx1[0]*k1,r1,q1,d1,input_signal)
  k3 = h*func(t+A[2]*h,y+Bx1[1]*k1+Bx2[0]*k2,r1,q1,d1,input_signal)
  k4 = h*func(t+A[3]*h,y+Bx1[2]*k1+Bx2[1]*k2+Bx3[0]*k3,r1,q1,d1,input_signal)
  k5 = h*func(t+A[4]*h,y+Bx1[3]*k1+Bx2[2]*k2+Bx3[1]*k3+Bx4[0]*k4,r1,q1,d1,input_signal)
  k6 = h*func(t+A[5]*h,y+Bx1[4]*k1+Bx2[3]*k2+Bx3[2]*k3+Bx4[1]*k4+Bx5*k5,r1,q1,d1,input_signal)
  return y+CH1*k1+CH2*k2+CH3*k3+CH4*k4+CH5*k5+CH6*k6
def my_RK45_time(t_eval,y_init,r1,q1,d1,input_signal):
  yt = y_init
  N = len(t_eval)
  y = yt
  for i in range(N-1):
    yt = my_RK45(t_eval[i],yt,t_eval[i+1]-t_eval[i],r1,q1,d1,input_signal)
    y = tf.concat([y,yt],0)
  return y  # Nx2

t_eval_ts = tf.convert_to_tensor(t_eval, dtype=tf.float32)
t_train_ts = tf.convert_to_tensor(t_train, dtype=tf.float32)
y_train_ts = tf.convert_to_tensor(y_train,dtype=tf.float32)
params = params_save = tf.Variable([-0.0005,10.1,2.1,0.1,0.1]) # r1,q1,d1, y_init_1,2

# TRAINING
alpha = 0.002 # learning rate 
iter = 0
iter_max = 100
eps = 1e-6
params_tmp = params
while iter < iter_max:
  iter = iter + 1
  with tf.GradientTape() as g:
    g.watch(params) 
    y = my_RK45_time(t_eval=t_train_ts,y_init=[params[3:]],r1=params[0],q1=params[1],d1=params[2],input_signal=input_signal)
    L = tf.reduce_mean((y[:,0]-y_train_ts)**2)
  dL_dp = g.gradient(L, params)
  params = params - alpha*dL_dp
  if abs(params[0]-params_tmp[0])<eps and abs(params[1]-params_tmp[1])<eps and abs(params[2]-params_tmp[2])<eps and abs(params[3]-params_tmp[3])<eps:
    break
  params_tmp = params
  print(*['Iteration',iter,':','(r1,q1,d1) = (',params[0].numpy(),params[1].numpy(),params[2].numpy(),');','w_init = ',params[3:].numpy()])

print(*['Initial params: (r1,q1,d1) = (',params_save[0].numpy(),params_save[1].numpy(),params_save[2].numpy(),');','w_init = ',params_save[3:].numpy()])
print(*['Ground truth params: (r1,q1,d1) = (',r10,q10,d10,');','w0 = ',y0])

# TESTING and RESULT
def show_results_train_test(input_signal="sin",N_plot=5000): # N_plot > N_train
  sol = solve_ivp(func_np, t_span=t_span, y0=y0,t_eval=t_eval,args=(r10,q10,d10,input_signal)) # redundance
  y_GT = sol.y[0]
  y = my_RK45_time(t_eval=t_eval_ts,y_init=[params[3:]],r1=params[0],q1=params[1],d1=params[2],input_signal=input_signal)
  fig, axs = plt.subplots(1,2)
  axs[0].plot(t_eval[0:N_plot],np.sin(t_eval)[0:N_plot])
  axs[0].set_title('Input signal')
  axs[0].set_xlabel('Time (s)')
  axs[1].plot(t_eval[0:N_plot],y_GT[0:N_plot], color = 'r')
  axs[1].plot(t_train_ts,y[:,0][0:N_train], color = 'g')
  axs[1].plot(t_eval_ts[N_train-1:N_plot],y[:,0][N_train-1:N_plot], color = 'b')
  plt.legend(['ground truth', 'predicted-train','predicted-test'], loc='upper right')
  axs[1].set_title('Output')
  axs[1].set_xlabel('Time (s)')

def show_results_test(input_signal="step",N_plot=5000):
  sol = solve_ivp(func_np, t_span=t_span, y0=y0,t_eval=t_eval,args=(r10,q10,d10,input_signal))
  y_GT = sol.y[0]
  y = my_RK45_time(t_eval=t_eval_ts,y_init=[params[3:]],r1=params[0],q1=params[1],d1=params[2],input_signal=input_signal)
  fig, axs = plt.subplots(1,2)
  if input_signal == "sin":
    axs[0].plot(t_eval[0:N_plot],np.sin(t_eval)[0:N_plot])
  elif input_signal == "step":
    axs[0].plot(t_eval[0:N_plot],1.0*np.ones(shape=(len(t_eval,)))[0:N_plot])
  elif input_signal == "ramp":
    axs[0].plot(t_eval[0:N_plot],ramp(t_eval,5.0)[0:N_plot])
  elif input_signal == "pulse":
    axs[0].plot(t_eval[0:N_plot],pulse(t_eval,5.0)[0:N_plot])
  axs[0].set_title('Input signal')
  axs[0].set_xlabel('Time (s)')
  axs[1].plot(t_eval[0:N_plot],y_GT[0:N_plot], color = 'r')
  axs[1].plot(t_eval_ts[0:N_plot],y[:,0][0:N_plot], color = 'b')
  plt.legend(['ground truth', 'predicted-test'], loc='upper right')
  axs[1].set_title('Output')
  axs[1].set_xlabel('Time (s)')

show_results_train_test(input_signal,len(t_eval)) # default: signal has been used for training
show_results_test("step",len(t_eval))
show_results_test("ramp",len(t_eval))
show_results_test("pulse",len(t_eval))
