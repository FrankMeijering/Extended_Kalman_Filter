import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def plot3dline(point1, point2, axis, colour='k', lbl=''):
    """
    Converts two points into a line and plots it on the given axis.

    :param colour: colour of the desired plot.
    :param point1: Starting coordinate.
    :param point2: Ending coordinate.
    :param axis: Figure axis to be plotted on.
    :param lbl: Label of the axis.
    :return: None
    """
    if lbl == '':
        axis.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c=colour)
    else:
        axis.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c=colour, label=lbl)


def plot3dframe(frame, axis, colour1='k', colour2='k', colour3='k', lbl1='', lbl2='', lbl3=''):
    """
    Plots a reference frame with three axes in a 3D plot.

    :param frame: 3x3 Numpy array containing the three unit vectors in the desired directions.
    :param axis: Matplotlib axis to be plotted on.
    :param colour1: Colour of the X-axis to be plotted.
    :param colour2: Colour of the Y-axis to be plotted.
    :param colour3: Colour of the Y-axis to be plotted.
    :param lbl1: Label of the X-axis to be shown in the legend.
    :param lbl2: Label of the Y-axis to be shown in the legend.
    :param lbl3: Label of the Z-axis to be shown in the legend.
    :return: None
    """
    origin = np.array([0, 0, 0])
    for i in range(3):
        if i == 0:
            plot3dline(origin, frame[i], axis, colour1, lbl1)
        elif i == 1:
            plot3dline(origin, frame[i], axis, colour2, lbl2)
        else:
            plot3dline(origin, frame[i], axis, colour3, lbl3)


def euler321(theta):
    """
    Performs the 3-2-1 rotation by calculating the individual rotations over the theta angles, and multiplying them.

    :param theta: 3-dimensional vector with three Euler angles (or an extended vector with the Euler angles on the first
     three elements). Order of rotation is 3, 2, 1.
    :return: Transformation matrix as a 3x3 Numpy array
    """
    C3 = np.array([[np.cos(theta[2]), np.sin(theta[2]), 0],
                   [-np.sin(theta[2]), np.cos(theta[2]), 0],
                   [0, 0, 1]])
    C2 = np.array([[np.cos(theta[1]), 0, -np.sin(theta[1])],
                   [0, 1, 0],
                   [np.sin(theta[1]), 0, np.cos(theta[1])]])
    C1 = np.array([[1, 0, 0],
                   [0, np.cos(theta[0]), np.sin(theta[0])],
                   [0, -np.sin(theta[0]), np.cos(theta[0])]])
    return C1 @ C2 @ C3


def dx_dt(t, x):
    """
    Calculates the derivatives of the 3-2-1 Euler angles, based on the body rotational rates.
    Needs the global variable omega

    :param t: time vector, is unused but necessary for scipy.integrate.solve_ivp
    :param x: state vector, consisting of the Euler angles theta and the biases on omega
    :return: derivative x_dot
    """
    x_dot = np.zeros(n)  # n is global variable
    multipl = np.array(
        [[np.cos(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[0]) * np.sin(x[1])],
         [0, np.cos(x[0]) * np.cos(x[1]), -np.sin(x[0]) * np.cos(x[1])],
         [0, np.sin(x[0]), np.cos(x[0])]])
    x_dot[:3] = 1 / np.cos(x[1]) * multipl @ (omega_meas - x[3:])
    x_dot[3:] = 0.
    return x_dot


def pd_ctrl(x, dxdt):
    """
    Calculates the control torque for a given input state, desiring an end state of zero.

    :param x: State vector (first three elements are theta)
    :param dxdt: Derivative function of the state vector
    :param dt: Timestep
    :param integ: Integral of all previous angles (are added each timestep)
    :return: Control torque [Nm]
    """
    # PD controller, acts only on theta (=x[:3])
    kp = np.diag(np.array([0.5730, 0.5730, 0.5730]))
    kd = np.diag(np.array([54.58, 51.34, 58.63]))
    return kp@x[:3] + kd@(dxdt(0, x)[:3])  # [Nm] control torque


def domega_dt(t, om):
    """
    Calculates the derivatives of the body rates.

    :param t: time vector, is unused but necessary for scipy.integrate.solve_ivp
    :param om: Body rate omega
    :return: derivative omega_dot
    """

    # kinematic relations
    Omega = np.array([[0., -om[2], om[1]],
                      [om[2], 0., -om[0]],
                      [-om[1], om[0], 0.]])

    return J_inv @ (Td_body - ctrl - Omega @ J @ om)  # uses global variables


def kalman(x_k_1k_1, P_k_1k_1):
    """
    Applies an Extended Kalman Filter to predict the actual state of the system based on the measurements and noise/bias
    information.

    :param x_k_1k_1: Previous iteration of the state vector
    :param P_k_1k_1: Previous iteration of the error covariance matrix
    :return: x_k_1k_1, P_k_1k_1: Next iterations
    """

    # 1) Prediction
    outp = solve_ivp(dx_dt, [ti, tf], x_k_1k_1, method='RK45')
    x_k_1k = outp.y[:, -1]  # x_{k+1, k}
    z_k_1k = sensors(x_k_1k)  # sensors and state are directly related

    F = f_jacobian(x_k_1k_1)
    Phi = expm(F * dt)
    Gamma = dt * np.array([[1., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0.]])
    P_k_1k = Phi @ P_k_1k_1 @ Phi.T + Gamma @ Q @ Gamma.T

    # 2) Correction
    H = h_jacobian(x_k_1k)
    Ve = H @ P_k_1k @ H.T + R  # Covariance of innovation
    K = P_k_1k @ H.T @ np.linalg.inv(Ve)
    x_kk = x_k_1k + K @ (z - z_k_1k)
    P_kk = (np.eye(n) - K @ H) @ P_k_1k @ (np.eye(n) - K @ H).T + K @ R @ K.T
    x_k_1k_1 = x_kk
    P_k_1k_1 = P_kk

    # Store data
    x_pred[timestep] = x_k_1k
    z_pred[timestep] = z_k_1k
    P_pred[timestep] = np.diag(P_k_1k)
    stdx_pred[timestep] = np.sqrt(np.diag(P_k_1k))

    Ve_k[timestep] = np.sqrt(np.diag(Ve))
    x_cor[timestep] = x_kk
    P_cor[timestep] = np.diag(P_kk)
    stdx_cor[timestep] = np.sqrt(np.diag(P_kk))

    return x_k_1k_1, P_k_1k_1


def f_jacobian(x):  # derivative of derivative func
    """
    Returns the F matrix, which is the Jacobian of the dx/dt function.

    :param x: State vector
    :return: F: 6x6 Jacobian matrix of the state vector
    """
    F = np.array([[np.cos(x[0])*np.tan(x[1])*(omega_meas[1]-x[4])-np.sin(x[0])*np.tan(x[1])*(omega_meas[2]-x[5]),
                   (np.sin(x[0])*(omega_meas[1]-x[4]) + np.cos(x[0])*(omega_meas[2]-x[5]))/(np.cos(x[1]))**2, 0., -1.,
                   -np.sin(x[0])*np.tan(x[1]), -np.cos(x[0])*np.tan(x[1])],
                  [-np.sin(x[0])*(omega_meas[1]-x[4])-np.cos(x[0])*(omega_meas[2]-x[5]), 0., 0., 0., -np.cos(x[0]),
                   np.sin(x[0])],
                  [np.cos(x[0])/np.cos(x[1])*(omega_meas[1]-x[4])-np.sin(x[0])/np.cos(x[1])*(omega_meas[2]-x[5]),
                   np.sin(x[1])/(np.cos(x[1]))**2*(np.sin(x[0])*(omega_meas[1]-x[4])+np.cos(x[0])*(omega_meas[2]-x[5])),
                   0., 0., -np.sin(x[0])/np.cos(x[1]), -np.cos(x[0])/np.cos(x[1])],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.]])
    return F


def sensors(x):
    """
    Mapping of the state vector onto the measurement vector.

    :param x: State vector, including both theta and bias
    :return: z: Measurement vector, including only theta
    """
    return x[:3]


def h_jacobian(x):
    """
    Returns the H matrix, which is the Jacobian of the sensors mapping; note the 3x6 shape.

    :param x: State vector
    :return: H: 3x6 Jacobian matrix
    """
    return np.array([[1., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0.]])


# ---------------------------------- User-Defined Parameters ---------------------------------------
# Time properties
dt = 0.2  # [s]
t_end = 2000.  # [s]

# Other
animate = True  # Display an animation (True) or not (False)
speed = 100  # [-] Animation speed, =1 includes all data. Increase value to skip data points and speed up the animation.
ctrl_on = True  # Apply PD controller (True), or let the spacecraft tumble by the disturbance torque only (False).
kalman_on = True  # Apply Kalman filter and use for the control loop (True), or only use the raw measurements (False).

# --------------------------------------- Define System ---------------------------------------------
# Define axis system
a = np.eye(3)  # a-frame is the reference frame, which is the local vertical local horizontal (LVLH) frame

# Spacecraft and environment characteristics
J11 = 2600.  # [kgm^2]
J22 = 2300.  # [kgm^2]
J33 = 3000.  # [kgm^2]
J = np.array([[J11, 0., 0.],
              [0., J22, 0.],
              [0., 0., J33]])  # [kgm^2] Satellite rotational moments of inertia
J_inv = np.linalg.inv(J)  # Invert it once now, to avoid having to invert it numerous times
Td = np.array([0.001, 0.001, 0.001])  # [Nm] Constant disturbance torque, in the LVLH frame

# Time properties
N_steps = int(t_end/dt)  # Number of time steps (length of the propagation)
time = np.arange(0, N_steps*dt, dt)  # [s] Time vector
ti = 0.
tf = dt

# Initial conditions
n = 6  # number of dimensions of the state vector
x = np.array([5., 5., 5., 0.1, -0.1, 0.15])*np.pi/180  # State vector
C = euler321(x)  # Direction cosine matrix for a 3-2-1 rotation with Euler angles x[:3]
omega = np.array([0., 0., 0.])  # [rad/s] Rotational rates in body frame
ctrl = np.zeros(3)  # [Nm] Initialise control torque at zero. Stays at zero if ctrl_on = False.

# Kalman filter parameters:
# 1) Noise and bias characteristics
stdw = np.array([0., 0., 0., 0., 0., 0.])*np.pi/180  # St. dev. of system noise on theta [rad] and bias [rad/s]
Q = np.diag(stdw**2)  # System noise covariance matrix
w_k = stdw*np.random.randn(N_steps, n)  # Actual noise applied to system

stdv = np.array([0.1, 0.1, 0.1])*np.pi/180  # [rad] St. dev. of measurement noise, for theta only
R = np.diag(stdv**2)  # Measurement noise covariance matrix
v_k = stdv*np.random.randn(N_steps, 3)  # Actual noise applied to the attitude measurements

omega_meas = omega + x[3:]  # [rad/s] Body rates measured by the gyros (including bias)

# 2) Initialise variables for the calculation loops
Ex_0 = np.array([0., 0., 0., 0., 0., 0.])*np.pi/180  # Expected initial value of theta [rad] and bias [rad/s]
stdx_0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])*np.pi/180  # Expected initial st. dev. of theta [rad] and bias [rad/s]
P_0 = np.diag(stdx_0**2)  # Expected initial error covariance matrix

P_k_1k_1 = P_0
if kalman_on:  # Use expected value for x_guess
    x_k_1k_1 = Ex_0
else:  # Use measured value for x_guess
    z = sensors(x) + v_k[0]
    x_k_1k_1 = np.concatenate((z, np.zeros(3)))  # Bias is unknown and used values for x_guess are measurements

# Initialise lists:
# 1) Actual parameters
x_list = np.zeros((N_steps, n))
z_list = np.zeros((N_steps, 3))
omega_list = np.zeros((N_steps, 3))
b1_list = np.zeros((N_steps, 3))
b2_list = np.zeros((N_steps, 3))
b3_list = np.zeros((N_steps, 3))
ctrl_list = np.zeros((N_steps, 3))
# 2) Kalman parameters (will remain zero if the Kalman filter is turned off)
x_pred = np.zeros((N_steps, n))
z_pred = np.zeros((N_steps, 3))
P_pred = np.zeros((N_steps, n))
stdx_pred = np.zeros((N_steps, n))
Ve_k = np.zeros((N_steps, 3))
x_cor = np.zeros((N_steps, n))
P_cor = np.zeros((N_steps, n))
stdx_cor = np.zeros((N_steps, n))

# ---------------------------------- Perform Calculations ---------------------------------------
print('---------- COMPUTING ----------')
for timestep in range(N_steps):
    if timestep != 0 and timestep % 500 == 0:
        print(f'{timestep/N_steps * 100:.2f} %')

    t = timestep*dt
    x += w_k[timestep]*dt  # Add system noise to the state

    Td_body = C @ Td  # Map the disturbance torque on the body frame

    if ctrl_on:  # Apply PD-controller
        ctrl = pd_ctrl(x_k_1k_1, dx_dt)

    # Integrate real state
    out = solve_ivp(dx_dt, [ti, tf], x, method='RK45')  # Integrate x
    x = out.y[:, -1]
    z = sensors(x) + v_k[timestep]

    out = solve_ivp(domega_dt, [ti, tf], omega, method='RK45')  # Integrate omega
    omega = out.y[:, -1]  # Extract data from the solver
    omega_meas = omega + x[3:]

    # Apply Kalman filter
    if kalman_on:
        # Use filter to make a prediction and update
        x_k_1k_1, P_k_1k_1 = kalman(x_k_1k_1, P_k_1k_1)
    else:
        # Use raw measurement data
        x_k_1k_1 = np.concatenate((z, np.zeros(3)))

    # Update
    ti = tf
    tf = ti + dt
    C = euler321(x)
    b = C @ a  # Store theta as the b axis system, for the animation

    # Store data
    ctrl_list[timestep] = -ctrl  # minus is added in case the torque is desired to be regarded as an "external" torque
    x_list[timestep] = x
    z_list[timestep] = z
    omega_list[timestep] = omega

    b1_list[timestep] = b[0]
    b2_list[timestep] = b[1]
    b3_list[timestep] = b[2]
print('---------- FINISHED ----------')

# ------------------------------------- Plotting ---------------------------------------
fig1 = plt.figure()
ax11 = fig1.add_subplot(3, 1, 1)
ax11.plot(time, z_list[:, 0]*180/np.pi, label=r'measurement')
ax11.plot(time, x_list[:, 0]*180/np.pi, label=r'actual')
if kalman_on:
    ax11.plot(time, x_pred[:, 0]*180/np.pi, label='prediction')
ax11.set_ylabel(r'Roll angle $\phi$ [$\degree$]')
ax11.legend()
ax12 = fig1.add_subplot(3, 1, 2)
ax12.plot(time, z_list[:, 1]*180/np.pi, label=r'measurement')
ax12.plot(time, x_list[:, 1]*180/np.pi, label=r'actual')
if kalman_on:
    ax12.plot(time, x_pred[:, 1]*180/np.pi, label='prediction')
ax12.set_ylabel(r'Pitch angle $\theta$ [$\degree$]')
ax12.legend()
ax13 = fig1.add_subplot(3, 1, 3)
ax13.plot(time, z_list[:, 2]*180/np.pi, label=r'measurement')
ax13.plot(time, x_list[:, 2]*180/np.pi, label=r'actual')
if kalman_on:
    ax13.plot(time, x_pred[:, 2]*180/np.pi, label='prediction')
ax13.set_ylabel(r'Yaw angle $\psi$ [$\degree$]')
ax13.set_xlabel('Time [s]')  # x-axis is shared
ax13.legend()
fig1.tight_layout()

if kalman_on:
    fig4 = plt.figure()
    ax41 = fig4.add_subplot(3, 2, 1)
    ax41.plot(time, x_pred[:, 0]*180/np.pi-x_list[:, 0]*180/np.pi)
    ax41.plot(time, stdx_pred[:, 0]*180/np.pi)
    ax41.plot(time, -stdx_pred[:, 0]*180/np.pi)
    ax41.set_ylabel(r'Error of $\phi$ [$\degree$]')
    ax42 = fig4.add_subplot(3, 2, 2)
    ax42.plot(time, x_pred[:, 3]*180/np.pi-x_list[:, 3]*180/np.pi)
    ax42.plot(time, stdx_pred[:, 3]*180/np.pi)
    ax42.plot(time, -stdx_pred[:, 3]*180/np.pi)
    ax42.set_ylabel(r'Error of bias $\dot{\phi}$ [$\degree$/s]')
    ax43 = fig4.add_subplot(3, 2, 3)
    ax43.plot(time, x_pred[:, 1]*180/np.pi-x_list[:, 1]*180/np.pi)
    ax43.plot(time, stdx_pred[:, 1]*180/np.pi)
    ax43.plot(time, -stdx_pred[:, 1]*180/np.pi)
    ax43.set_ylabel(r'Error of $\theta$ [$\degree$]')
    ax44 = fig4.add_subplot(3, 2, 4)
    ax44.plot(time, x_pred[:, 4]*180/np.pi-x_list[:, 4]*180/np.pi)
    ax44.plot(time, stdx_pred[:, 4]*180/np.pi)
    ax44.plot(time, -stdx_pred[:, 4]*180/np.pi)
    ax44.set_ylabel(r'Error of bias $\dot{\theta}$ [$\degree$/s]')
    ax45 = fig4.add_subplot(3, 2, 5)
    ax45.plot(time, x_pred[:, 2]*180/np.pi-x_list[:, 2]*180/np.pi)
    ax45.plot(time, stdx_pred[:, 2]*180/np.pi)
    ax45.plot(time, -stdx_pred[:, 2]*180/np.pi)
    ax45.set_ylabel(r'Error of $\psi$ [$\degree$]')
    ax45.set_xlabel('Time [s]')
    ax46 = fig4.add_subplot(3, 2, 6)
    ax46.plot(time, x_pred[:, 5]*180/np.pi-x_list[:, 5]*180/np.pi)
    ax46.plot(time, stdx_pred[:, 5]*180/np.pi)
    ax46.plot(time, -stdx_pred[:, 5]*180/np.pi)
    ax46.set_ylabel(r'Error of bias $\dot{\psi}$ [$\degree$/s]')
    ax46.set_xlabel('Time [s]')
    fig4.tight_layout()

    fig2 = plt.figure()
    ax21 = fig2.add_subplot(3, 1, 1)
    ax21.plot(time, x_list[:, 3] * 180 / np.pi, label=r'actual')
    ax21.plot(time, x_pred[:, 3] * 180 / np.pi, label='prediction')
    ax21.set_ylabel(r'd$\phi$/dt bias [$\degree$/s]')
    ax21.legend()
    ax22 = fig2.add_subplot(3, 1, 2)
    ax22.plot(time, x_list[:, 4] * 180 / np.pi, label=r'actual')
    ax22.plot(time, x_pred[:, 4] * 180 / np.pi, label='prediction')
    ax22.set_ylabel(r'd$\theta$/dt bias [$\degree$/s]')
    ax22.legend()
    ax23 = fig2.add_subplot(3, 1, 3)
    ax23.plot(time, x_list[:, 5] * 180 / np.pi, label=r'actual')
    ax23.plot(time, x_pred[:, 5] * 180 / np.pi, label='prediction')
    ax23.set_ylabel(r'd$\psi$/dt bias [$\degree$/s]')
    ax23.set_xlabel('Time [s]')
    ax23.legend()
    fig2.tight_layout()

if ctrl_on:
    fig3 = plt.figure()
    ax31 = fig3.add_subplot(3, 1, 1)
    ax31.plot(time, ctrl_list[:, 0])
    ax31.set_ylabel(r'Roll torque $T_\phi$ [Nm]')
    ax32 = fig3.add_subplot(3, 1, 2)
    ax32.plot(time, ctrl_list[:, 1])
    ax32.set_ylabel(r'Pitch torque $T_\theta$ [Nm]')
    ax33 = fig3.add_subplot(3, 1, 3)
    ax33.plot(time, ctrl_list[:, 2])
    ax33.set_ylabel(r'Yaw torque $T_\psi$ [Nm]')
    ax33.set_xlabel(r'Time [s]')  # x-axes are shared
    fig3.tight_layout()

plt.show(block=True)  # block=True indicates that all windows have to be closed before the code is continued.

# ---------------------------------------------- Animate ------------------------------------------
if animate:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    timestep = 0
    while plt.fignum_exists(fig1.number) and timestep < N_steps:  # Use fignum to be able to close window normally
        if timestep % speed == 0:
            ax1.clear()
            # seconds = timestep*dt
            days = timestep * dt // (24 * 3600)
            hours = (timestep * dt % (24 * 3600)) // 3600
            minutes = (timestep * dt % 3600) // 60
            seconds = (timestep * dt % 60)
            plt.suptitle(f'Time passed: {hours:.0f} hrs, {minutes:.0f} mins, {seconds:.2f} seconds')

            b = np.vstack((b1_list[timestep], b2_list[timestep], b3_list[timestep]))

            plot3dframe(a, ax1, 'firebrick', 'g', 'k', lbl1='Velocity', lbl2='Orbit Normal', lbl3='Nadir')
            plot3dframe(b, ax1, 'r', 'lime', 'b', lbl1='X-body', lbl2='Y-body', lbl3='Z-body')

            ax1.axes.set_xlim3d(left=-1, right=1)
            ax1.axes.set_ylim3d(bottom=-1, top=1)
            ax1.axes.set_zlim3d(bottom=-1, top=1)
            ax1.legend()

            plt.pause(0.0001)  # with a really fast computer, this can be set to dt for real-time
        timestep += 1
    plt.show(block=True)
