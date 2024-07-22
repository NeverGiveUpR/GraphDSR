import numpy as np

which_sample = 'random' # linspace

def I_6_2a(n=100):
    a, b = [1, 3]
    if which_sample == 'random':
        theta = (b-a)*np.random.random_sample(n) + a
    else:
        theta = np.linspace(a, b, n)
    y = np.exp(-theta**2/2)/np.sqrt(2*np.pi)
    X = theta[:, np.newaxis]
    return X, y

def I_6_20(n=100):
    a, b = [1, 3]
    theta = (b-a)*np.random.random_sample(n) + a
    sigma = (b-a)*np.random.random_sample(n) + a
    y = np.exp(-(theta/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)
    X = np.hstack((theta[:, np.newaxis], sigma[:, np.newaxis]))
    return X, y

def I_6_2b(n=1000):
    a, b = [1, 3]
    theta = (b-a)*np.random.random_sample(n) + a
    theta1 = (b-a)*np.random.random_sample(n) + a
    sigma = (b-a)*np.random.random_sample(n) + a
    X = np.hstack((theta[:, np.newaxis], sigma[:, np.newaxis]))
    X = np.hstack((X, theta1[:, np.newaxis]))
    y = np.exp(-((theta-theta1)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)
    return X, y

def I_8_14(n=10**3):
    a, b = [1, 5]
    # x1 = (b-a)*np.random.random_sample(n) + a
    # x2 = (b-a)*np.random.random_sample(n) + a
    # y1 = (b-a)*np.random.random_sample(n) + a
    # y2 = (b-a)*np.random.random_sample(n) + a
    x1 = np.linspace(a, b, n)
    np.random.shuffle(x1)
    x2 = np.linspace(a, b, n)
    np.random.shuffle(x2)
    y1 = np.linspace(a, b, n)
    np.random.shuffle(y1)
    y2 = np.linspace(a, b, n)
    np.random.shuffle(y2)
    y = np.sqrt((x2-x1)**2+(y2-y1)**2)
    X = np.hstack((x1[:, np.newaxis], x2[:, np.newaxis]))
    X = np.hstack((X, y1[:, np.newaxis]))
    X = np.hstack((X, y2[:, np.newaxis]))
    return X, y

def I_9_18(n=10**6):
    a, b = [1, 2]
    # m1 = (b-a)*np.random.random_sample(n) + a
    # m2 = (b-a)*np.random.random_sample(n) + a
    # G = (b-a)*np.random.random_sample(n) + a
    # x2 = (b-a)*np.random.random_sample(n) + a
    # y2 = (b-a)*np.random.random_sample(n) + a
    # z2 = (b-a)*np.random.random_sample(n) + a
    m1 = np.linspace(a, b, n)
    np.random.shuffle(m1)
    m2 = np.linspace(a, b, n)
    np.random.shuffle(m2)
    G = np.linspace(a, b, n)
    np.random.shuffle(G)
    x2 = np.linspace(a, b, n)
    np.random.shuffle(x2)
    y2 = np.linspace(a, b, n)
    np.random.shuffle(y2)
    z2 = np.linspace(a, b, n)
    np.random.shuffle(z2)
    a, b = [3, 4]
    # x1 = (b-a)*np.random.random_sample(n) + a
    # y1 = (b-a)*np.random.random_sample(n) + a
    # z1 = (b-a)*np.random.random_sample(n) + a
    x1 = np.linspace(a, b, n)
    np.random.shuffle(x1)
    y1 = np.linspace(a, b, n)
    np.random.shuffle(y1)
    z1 = np.linspace(a, b, n)
    np.random.shuffle(z1)
    y = G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    X = np.hstack((m1[:, np.newaxis], m2[:, np.newaxis]))
    X = np.hstack((X, G[:, np.newaxis]))
    X = np.hstack((X, x2[:, np.newaxis]))
    X = np.hstack((X, y2[:, np.newaxis]))
    X = np.hstack((X, z2[:, np.newaxis]))
    X = np.hstack((X, x1[:, np.newaxis]))
    X = np.hstack((X, y1[:, np.newaxis]))
    X = np.hstack((X, z1[:, np.newaxis]))
    return X, y

def I_10_7(n=10):
    a, b = [1, 5]
    # m_0 = (b-a)*np.random.random_sample(n) + a
    m_0 = np.linspace(a, b, n)
    np.random.shuffle(m_0)
    a, b = [1, 2]
    # v = (b-a)*np.random.random_sample(n) + a
    v = np.linspace(a, b, n)
    np.random.shuffle(v)
    a, b = [3, 10]
    # c = (b-a)*np.random.random_sample(n) + a
    c = np.linspace(a, b, n)
    np.random.shuffle(c)
    y = m_0/np.sqrt(1-v**2/c**2)
    X = np.hstack((m_0[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    return X, y

def I_11_19(n=100):
    a, b = [1, 5]
    x1 = (b-a)*np.random.random_sample(n) + a
    x2 = (b-a)*np.random.random_sample(n) + a
    x3 = (b-a)*np.random.random_sample(n) + a
    y1 = (b-a)*np.random.random_sample(n) + a
    y2 = (b-a)*np.random.random_sample(n) + a
    y3 = (b-a)*np.random.random_sample(n) + a
    y = x1*y1+x2*y2+x3*y3
    X = np.hstack((x1[:, np.newaxis], x2[:, np.newaxis]))
    X = np.hstack((X, x3[:, np.newaxis]))
    X = np.hstack((X, y1[:, np.newaxis]))
    X = np.hstack((X, y2[:, np.newaxis]))
    X = np.hstack((X, y3[:, np.newaxis]))
    return X, y

def I_12_1(n=10):
    a, b = [1, 5]
    mu = (b-a)*np.random.random_sample(n) + a
    Nn = (b-a)*np.random.random_sample(n) + a
    y = mu*Nn
    X = np.hstack((mu[:, np.newaxis], Nn[:, np.newaxis]))
    return X, y

def I_12_2(n=10):
    a, b = [1, 5]
    q1 = (b-a)*np.random.random_sample(n) + a
    q2 = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    y = q1*q2*r/(4*np.pi*epsilon*r**3)
    X = np.hstack((q1[:, np.newaxis], q2[:, np.newaxis]))
    X = np.hstack((X, r[:, np.newaxis]))
    X = np.hstack((X, epsilon[:, np.newaxis]))
    return X, y

def I_12_4(n=10):
    a, b = [1, 5]
    q1 = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    X = np.hstack((q1[:, np.newaxis], r[:, np.newaxis]))
    X = np.hstack((X, epsilon[:, np.newaxis]))
    y = q1*r/(4*np.pi*epsilon*r**3)
    return X, y

def I_12_5(n=10):
    a, b = [1, 5]
    q2 = (b-a)*np.random.random_sample(n) + a
    E_f = (b-a)*np.random.random_sample(n) + a
    y = q2*E_f
    X = np.hstack((q2[:, np.newaxis], E_f[:, np.newaxis]))
    return X, y

def I_12_11(n=10):
    a, b = [1, 5]
    q = (b-a)*np.random.random_sample(n) + a
    Ef = (b-a)*np.random.random_sample(n) + a
    B = (b-a)*np.random.random_sample(n) + a
    v = (b-a)*np.random.random_sample(n) + a
    theta = (b-a)*np.random.random_sample(n) + a
    y = q*(Ef+B*v*np.sin(theta))
    X = np.hstack((q[:, np.newaxis], Ef[:, np.newaxis]))
    X = np.hstack((X, B[:, np.newaxis]))
    X = np.hstack((X, v[:, np.newaxis]))
    X = np.hstack((X, theta[:, np.newaxis]))
    return X, y

def I_13_4(n=10):
    a, b = [1, 5]
    m = (b-a)*np.random.random_sample(n) + a
    v = (b-a)*np.random.random_sample(n) + a
    u = (b-a)*np.random.random_sample(n) + a
    w = (b-a)*np.random.random_sample(n) + a
    y = 1/2*m*(v**2+u**2+w**2)
    X = np.hstack((m[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, u[:, np.newaxis]))
    X = np.hstack((X, w[:, np.newaxis]))
    return X, y

def I_13_12(n=10):
    a, b = [1, 5]
    G = (b-a)*np.random.random_sample(n) + a
    m1 = (b-a)*np.random.random_sample(n) + a
    m2 = (b-a)*np.random.random_sample(n) + a
    r1 = (b-a)*np.random.random_sample(n) + a
    r2 = (b-a)*np.random.random_sample(n) + a
    y = G*m1*m2*(1/r2-1/r1)
    X = np.hstack((G[:, np.newaxis], m1[:, np.newaxis]))
    X = np.hstack((X, m2[:, np.newaxis]))
    X = np.hstack((X, r1[:, np.newaxis]))
    X = np.hstack((X, r2[:, np.newaxis]))
    return X, y

def I_14_3(n=10):
    a, b = [1, 5]
    m = (b-a)*np.random.random_sample(n) + a
    g = (b-a)*np.random.random_sample(n) + a
    z = (b-a)*np.random.random_sample(n) + a
    y = m*g*z
    X = np.hstack((m[:, np.newaxis], g[:, np.newaxis]))
    X = np.hstack((X, z[:, np.newaxis]))
    return X, y

def I_14_4(n=10):
    a, b = [1, 5]
    k_spring = (b-a)*np.random.random_sample(n) + a
    x = (b-a)*np.random.random_sample(n) + a
    y = 1/2*k_spring*x**2
    X = np.hstack((k_spring[:, np.newaxis], x[:, np.newaxis]))
    return X, y

def I_15_3x(n=10):
    a, b = [5, 10]
    x = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    u = (b-a)*np.random.random_sample(n) + a
    a, b = [3, 20]
    c = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    t = (b-a)*np.random.random_sample(n) + a
    y = (x-u*t)/np.sqrt(1-u**2/c**2)
    X = np.hstack((x[:, np.newaxis], u[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    X = np.hstack((X, t[:, np.newaxis]))
    return X, y

def I_15_3t(n=100):
    a, b = [1, 5]
    x = (b-a)*np.random.random_sample(n) + a
    a, b = [3, 10]
    c = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    u = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 5]
    t = (b-a)*np.random.random_sample(n) + a
    y = (t-u*x/c**2)/np.sqrt(1-u**2/c**2)
    X = np.hstack((x[:, np.newaxis], c[:, np.newaxis]))
    X = np.hstack((X, u[:, np.newaxis]))
    X = np.hstack((X, t[:, np.newaxis]))
    return X, y

def I_15_10(n=100):
    a, b = [1, 5]
    m_0 = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    v = (b-a)*np.random.random_sample(n) + a
    a, b = [3, 10]
    c = (b-a)*np.random.random_sample(n) + a
    y = m_0*v/np.sqrt(1-v**2/c**2)
    X = np.hstack((m_0[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    return X, y

def I_16_6(n=10):
    a, b = [1, 5]
    u = (b-a)*np.random.random_sample(n) + a
    v = (b-a)*np.random.random_sample(n) + a
    c = (b-a)*np.random.random_sample(n) + a
    y = (u+v)/(1+u*v/c**2)
    X = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    return X, y

def I_18_4(n=10):
    a, b = [1, 5]
    m1 = (b-a)*np.random.random_sample(n) + a
    m2 = (b-a)*np.random.random_sample(n) + a
    r1 = (b-a)*np.random.random_sample(n) + a
    r2 = (b-a)*np.random.random_sample(n) + a
    y = (m1*r1+m2*r2)/(m1+m2)
    X = np.hstack((m1[:, np.newaxis], m2[:, np.newaxis]))
    X = np.hstack((X, r1[:, np.newaxis]))
    X = np.hstack((X, r2[:, np.newaxis]))
    return X, y

def I_18_12(n=10):
    a, b = [1, 5]
    r = (b-a)*np.random.random_sample(n) + a
    F = (b-a)*np.random.random_sample(n) + a
    a, b = [0, 5]
    theta = (b-a)*np.random.random_sample(n) + a
    y = r*F*np.sin(theta)
    X = np.hstack((r[:, np.newaxis], F[:, np.newaxis]))
    X = np.hstack((X, theta[:, np.newaxis]))
    return X, y

def I_18_14(n=10):
    a, b = [1, 5]
    m = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    v = (b-a)*np.random.random_sample(n) + a
    theta = (b-a)*np.random.random_sample(n) + a
    y = m*r*v*np.sin(theta)
    X = np.hstack((m[:, np.newaxis], r[:, np.newaxis]))
    X = np.hstack((X, v[:, np.newaxis]))
    X = np.hstack((X, theta[:, np.newaxis]))
    return X, y

def I_24_6(n=10):
    a, b = [1, 3]
    m = (b-a)*np.random.random_sample(n) + a
    omega = (b-a)*np.random.random_sample(n) + a
    omega_0 = (b-a)*np.random.random_sample(n) + a
    x = (b-a)*np.random.random_sample(n) + a
    y = 1/2*m*(omega**2+omega_0**2)*1/2*x**2
    X = np.hstack((m[:, np.newaxis], omega[:, np.newaxis]))
    X = np.hstack((X, omega_0[:, np.newaxis]))
    X = np.hstack((X, x[:, np.newaxis]))
    return X, y

def I_25_13(n=10):
    a, b = [1, 3]
    q = (b-a)*np.random.random_sample(n) + a
    I = (b-a)*np.random.random_sample(n) + a
    y = q/I
    X = np.hstack((q[:, np.newaxis], I[:, np.newaxis]))
    return X, y

def I_26_2(num=100):
    a, b = [0, 1]
    n = (b-a)*np.random.random_sample(num) + a
    a, b = [1, 5]
    theta2 = (b-a)*np.random.random_sample(num) + a
    y = np.arcsin(n*np.sin(theta2))
    X = np.hstack((n[:, np.newaxis], theta2[:, np.newaxis]))
    return X, y

def I_27_6(num=10):
    a, b = [1, 5]
    d1 = (b-a)*np.random.random_sample(num) + a
    n = (b-a)*np.random.random_sample(num) + a
    d2 = (b-a)*np.random.random_sample(num) + a
    y = 1/(1/d1+n/d2)
    X = np.hstack((d1[:, np.newaxis], n[:, np.newaxis]))
    X = np.hstack((X, d2[:, np.newaxis]))
    return X, y

def I_29_4(n=10):
    a, b = [1, 10]
    omega = (b-a)*np.random.random_sample(n) + a
    c = (b-a)*np.random.random_sample(n) + a
    y = omega/c
    X = np.hstack((omega[:, np.newaxis], c[:, np.newaxis]))
    return X, y

def I_29_16(n=10**3):
    a, b = [1, 5]
    x1 = (b-a)*np.random.random_sample(n) + a
    x2 = (b-a)*np.random.random_sample(n) + a
    theta1 = (b-a)*np.random.random_sample(n) + a
    theta2 = (b-a)*np.random.random_sample(n) + a
    y = np.sqrt(x1**2+x2**2-2*x1*x2*np.cos(theta1-theta2))
    X = np.hstack((x1[:, np.newaxis], x2[:, np.newaxis]))
    X = np.hstack((X, theta1[:, np.newaxis]))
    X = np.hstack((X, theta2[:, np.newaxis]))
    return X, y

def I_30_3(num=10**2):
    a, b = [1, 5]
    Int_0 = (b-a)*np.random.random_sample(num) + a
    theta = (b-a)*np.random.random_sample(num) + a
    n = (b-a)*np.random.random_sample(num) + a
    y = Int_0*np.sin(n*theta/2)**2/np.sin(theta/2)**2
    X = np.hstack((Int_0[:, np.newaxis], theta[:, np.newaxis]))
    X = np.hstack((X, n[:, np.newaxis]))
    return X, y

def I_30_5(num=10**2):
    a, b = [1, 2]
    lambd = (b-a)*np.random.random_sample(num) + a
    a, b = [2, 5]
    d = (b-a)*np.random.random_sample(num) + a
    a, b = [1, 5]
    n = (b-a)*np.random.random_sample(num) + a
    y = np.arcsin(lambd/(n*d))
    X = np.hstack((lambd[:, np.newaxis], d[:, np.newaxis]))
    X = np.hstack((X, n[:, np.newaxis]))
    return X, y

def I_32_5(n=10):
    a, b = [1, 5]
    q = (b-a)*np.random.random_sample(n) + a
    a_ = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    c = (b-a)*np.random.random_sample(n) + a
    y = q**2*a_**2/(6*np.pi*epsilon*c**3)
    X = np.hstack((q[:, np.newaxis], a_[:, np.newaxis]))
    X = np.hstack((X, epsilon[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    return X, y

def I_32_17(n=10):
    a, b = [1, 2]
    epsilon = (b-a)*np.random.random_sample(n) + a
    c = (b-a)*np.random.random_sample(n) + a
    Ef = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    omega = (b-a)*np.random.random_sample(n) + a
    a, b = [3, 5]
    omega_0 = (b-a)*np.random.random_sample(n) + a
    y = (1/2*epsilon*c*Ef**2)*(8*np.pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)
    X = np.hstack((epsilon[:, np.newaxis], c[:, np.newaxis]))
    X = np.hstack((X, Ef[:, np.newaxis]))
    X = np.hstack((X, r[:, np.newaxis]))
    X = np.hstack((X, omega[:, np.newaxis]))
    X = np.hstack((X, omega_0[:, np.newaxis]))
    return X, y

def I_34_8(n=10):
    a, b = [1, 5]
    q = (b-a)*np.random.random_sample(n) + a
    v = (b-a)*np.random.random_sample(n) + a
    B = (b-a)*np.random.random_sample(n) + a
    p = (b-a)*np.random.random_sample(n) + a
    y = q*v*B/p
    X = np.hstack((q[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, B[:, np.newaxis]))
    X = np.hstack((X, p[:, np.newaxis]))
    return X, y

def I_34_10(n=10):
    a, b = [3, 10]
    omega_0 = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    v = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 5]
    c = (b-a)*np.random.random_sample(n) + a
    y = omega_0/(1-v/c)
    X = np.hstack((omega_0[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    return X, y

def I_34_14(n=10):
    a, b = [3, 10]
    omega_0 = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    v = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 5]
    c = (b-a)*np.random.random_sample(n) + a
    X = np.hstack((omega_0[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    y = (1+v/(c+1e-6))/np.sqrt(1-v**2/(c**2+1e-6))*omega_0
    return X, y

def I_34_27(n=10):
    a, b = [1, 5]
    h = (b-a)*np.random.random_sample(n) + a
    omega = (b-a)*np.random.random_sample(n) + a
    y = (h/(2*np.pi))*omega
    X = np.hstack((h[:, np.newaxis], omega[:, np.newaxis]))
    return X, y

def I_37_4(n=100):
    a, b = [1, 5]
    I1 = (b-a)*np.random.random_sample(n) + a
    I2 = (b-a)*np.random.random_sample(n) + a
    delta = (b-a)*np.random.random_sample(n) + a
    y = I1+I2+2*np.sqrt(I1*I2)*np.cos(delta)
    X = np.hstack((I1[:, np.newaxis], I2[:, np.newaxis]))
    X = np.hstack((X, delta[:, np.newaxis]))
    return X, y

def I_38_12(n=10):
    a, b = [1, 5]
    m = (b-a)*np.random.random_sample(n) + a
    q = (b-a)*np.random.random_sample(n) + a
    h = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    y = 4*np.pi*epsilon*(h/(2*np.pi))**2/(m*q**2)
    X = np.hstack((m[:, np.newaxis], q[:, np.newaxis]))
    X = np.hstack((X, h[:, np.newaxis]))
    X = np.hstack((X, epsilon[:, np.newaxis]))
    return X, y

def I_39_10(n=10):
    a, b = [1, 5]
    pr = (b-a)*np.random.random_sample(n) + a
    V = (b-a)*np.random.random_sample(n) + a
    y = 3/2*pr*V
    X = np.hstack((pr[:, np.newaxis], V[:, np.newaxis]))
    return X, y

def I_39_11(n=10):
    a, b = [2, 5]
    gamma = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 5]
    pr = (b-a)*np.random.random_sample(n) + a
    a, b = [2, 5]
    V = (b-a)*np.random.random_sample(n) + a
    y = 1/(gamma-1)*pr*V
    X = np.hstack((gamma[:, np.newaxis], pr[:, np.newaxis]))
    X = np.hstack((X, V[:, np.newaxis]))
    return X, y

def I_39_22(num=10):
    a, b = [1, 5]
    n = (b-a)*np.random.random_sample(num) + a
    kb = (b-a)*np.random.random_sample(num) + a
    T = (b-a)*np.random.random_sample(num) + a
    V = (b-a)*np.random.random_sample(num) + a
    y = n*kb*T/V
    X = np.hstack((n[:, np.newaxis], kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    X = np.hstack((X, V[:, np.newaxis]))
    return X, y

def I_40_10(n=10):
    a, b = [1, 5]
    n_0 = (b-a)*np.random.random_sample(n) + a
    m = (b-a)*np.random.random_sample(n) + a
    g = (b-a)*np.random.random_sample(n) + a
    x = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    y = n_0*np.exp(-m*g*x/(kb*T))
    X = np.hstack((n_0[:, np.newaxis], m[:, np.newaxis]))
    X = np.hstack((X, g[:, np.newaxis]))
    X = np.hstack((X, x[:, np.newaxis]))
    X = np.hstack((X, kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    return X, y

def I_41_16(n=10):
    a, b = [1, 5]
    h = (b-a)*np.random.random_sample(n) + a
    omega = (b-a)*np.random.random_sample(n) + a
    c = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    y = h/(2*np.pi)*omega**3/(np.pi**2*c**2*(np.exp((h/(2*np.pi))*omega/(kb*T))-1))
    X = np.hstack((h[:, np.newaxis], omega[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    X = np.hstack((X, kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    return X, y

def I_43_16(n=10):
    a, b = [1, 5]
    mu_drift = (b-a)*np.random.random_sample(n) + a
    q = (b-a)*np.random.random_sample(n) + a
    Volt = (b-a)*np.random.random_sample(n) + a
    d = (b-a)*np.random.random_sample(n) + a
    y =mu_drift*q*Volt/d
    X = np.hstack((mu_drift[:, np.newaxis], q[:, np.newaxis]))
    X = np.hstack((X, Volt[:, np.newaxis]))
    X = np.hstack((X, d[:, np.newaxis]))
    return X, y

def I_43_31(n=10):
    a, b = [1, 5]
    mob = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    X = np.hstack((mob[:, np.newaxis], kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    y = mob*kb*T
    return X, y

def I_43_43(n=10):
    a, b = [1, 5]
    gamma = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    v = (b-a)*np.random.random_sample(n) + a
    A = (b-a)*np.random.random_sample(n) + a
    y = 1/(gamma-1)*kb*v/A
    X = np.hstack((gamma[:, np.newaxis], kb[:, np.newaxis]))
    X = np.hstack((X, v[:, np.newaxis]))
    X = np.hstack((X, A[:, np.newaxis]))
    return X, y

def I_44_4(num=10):
    a, b = [1, 5]
    n = (b-a)*np.random.random_sample(num) + a
    kb = (b-a)*np.random.random_sample(num) + a
    T = (b-a)*np.random.random_sample(num) + a
    V2 = (b-a)*np.random.random_sample(num) + a
    V1 = (b-a)*np.random.random_sample(num) + a
    X = np.hstack((n[:, np.newaxis], kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    X = np.hstack((X, V2[:, np.newaxis]))
    X = np.hstack((X, V1[:, np.newaxis]))
    y = n*kb*T*np.log(V2/V1)
    return X, y

def I_47_23(n=10):
    a, b = [1, 5]
    gamma = (b-a)*np.random.random_sample(n) + a
    pr = (b-a)*np.random.random_sample(n) + a
    rho = (b-a)*np.random.random_sample(n) + a
    y = np.sqrt(gamma*pr/rho)
    X = np.hstack((gamma[:, np.newaxis], pr[:, np.newaxis]))
    X = np.hstack((X, rho[:, np.newaxis]))
    return X, y

def I_48_20(n=100):
    a, b = [1, 5]
    m = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    c = (b-a)*np.random.random_sample(n) + a
    a, b = [3, 10]
    v = (b-a)*np.random.random_sample(n) + a
    y = m*c**2/(np.sqrt(1-v**2/(c**2+1e-6))+1e-6)
    X = np.hstack((m[:, np.newaxis], c[:, np.newaxis]))
    X = np.hstack((X, v[:, np.newaxis]))
    return X, y

def I_50_26(n=10):
    a, b = [1, 3]
    x1 = (b-a)*np.random.random_sample(n) + a
    omega = (b-a)*np.random.random_sample(n) + a
    t = (b-a)*np.random.random_sample(n) + a
    alpha = (b-a)*np.random.random_sample(n) + a
    y = x1*(np.cos(omega*t)+alpha*np.cos(omega*t)**2)
    X = np.hstack((x1[:, np.newaxis], omega[:, np.newaxis]))
    X = np.hstack((X, t[:, np.newaxis]))
    X = np.hstack((X, alpha[:, np.newaxis]))
    return X, y

def II_2_42(n=10):
    a, b = [1, 5]
    kappa = (b-a)*np.random.random_sample(n) + a
    T2 = (b-a)*np.random.random_sample(n) + a
    T1 = (b-a)*np.random.random_sample(n) + a
    A = (b-a)*np.random.random_sample(n) + a
    d = (b-a)*np.random.random_sample(n) + a
    y = kappa*(T2-T1)*A/d
    X = np.hstack((kappa[:, np.newaxis], T2[:, np.newaxis]))
    X = np.hstack((X, T1[:, np.newaxis]))
    X = np.hstack((X, A[:, np.newaxis]))
    X = np.hstack((X, d[:, np.newaxis]))
    return X, y

def II_3_24(n=10):
    a, b = [1, 5]
    Pwr = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    y = Pwr/(4*np.pi*r**2)
    X = np.hstack((Pwr[:, np.newaxis], r[:, np.newaxis]))
    return X, y

def II_4_23(n=10):
    a, b = [1, 5]
    q = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    y = q/(4*np.pi*epsilon*r)
    X = np.hstack((q[:, np.newaxis], epsilon[:, np.newaxis]))
    X = np.hstack((X, r[:, np.newaxis]))
    return X, y

def II_6_11(n=10):
    a, b = [1, 3]
    epsilon = (b-a)*np.random.random_sample(n) + a
    p_d = (b-a)*np.random.random_sample(n) + a
    theta = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    y = 1/(4*np.pi*epsilon)*p_d*np.cos(theta)/r**2
    X = np.hstack((epsilon[:, np.newaxis], p_d[:, np.newaxis]))
    X = np.hstack((X, theta[:, np.newaxis]))
    X = np.hstack((X, r[:, np.newaxis]))
    return X, y

def II_6_15a(n=10**4):
    a, b = [1, 3]
    epsilon = (b-a)*np.random.random_sample(n) + a
    p_d = (b-a)*np.random.random_sample(n) + a
    z = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    x = (b-a)*np.random.random_sample(n) + a
    y = (b-a)*np.random.random_sample(n) + a
    output = p_d/(4*np.pi*epsilon)*3*z/r**5*np.sqrt(x**2+y**2)
    X = np.hstack((epsilon[:, np.newaxis], p_d[:, np.newaxis]))
    X = np.hstack((X, z[:, np.newaxis]))
    X = np.hstack((X, r[:, np.newaxis]))
    X = np.hstack((X, x[:, np.newaxis]))
    return X, output

def II_6_15b(n=10):
    a, b = [1, 3]
    epsilon = (b-a)*np.random.random_sample(n) + a
    p_d = (b-a)*np.random.random_sample(n) + a
    theta = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    X = np.hstack((epsilon[:, np.newaxis], p_d[:, np.newaxis]))
    X = np.hstack((X, theta[:, np.newaxis]))
    X = np.hstack((X, r[:, np.newaxis]))
    y = p_d/(4*np.pi*epsilon)*3*np.cos(theta)*np.sin(theta)/r**3
    return X, y

def II_8_7(n=10):
    a, b = [1, 5]
    q = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    d = (b-a)*np.random.random_sample(n) + a
    y = 3/5*q**2/(4*np.pi*epsilon*d)
    X = np.hstack((q[:, np.newaxis], epsilon[:, np.newaxis]))
    X = np.hstack((X, d[:, np.newaxis]))
    return X, y

def II_8_31(n=10):
    a, b = [1, 5]
    Ef = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    y = epsilon*Ef**2/2
    X = np.hstack((Ef[:, np.newaxis], epsilon[:, np.newaxis]))
    return X, y

def II_10_9(n=10):
    a, b = [1, 5]
    sigma_den = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    chi = (b-a)*np.random.random_sample(n) + a
    y = sigma_den/epsilon*1/(1+chi)
    X = np.hstack((sigma_den[:, np.newaxis], epsilon[:, np.newaxis]))
    X = np.hstack((X, chi[:, np.newaxis]))
    return X, y

def II_10_3(n=10):
    a, b = [1, 3]
    q = (b-a)*np.random.random_sample(n) + a
    Ef = (b-a)*np.random.random_sample(n) + a
    m = (b-a)*np.random.random_sample(n) + a
    a, b = [3, 5]
    omega_0 = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    omega = (b-a)*np.random.random_sample(n) + a
    y = q*Ef/(m*(omega_0**2-omega**2))
    X = np.hstack((q[:, np.newaxis], Ef[:, np.newaxis]))
    X = np.hstack((X, m[:, np.newaxis]))
    X = np.hstack((X, omega_0[:, np.newaxis]))
    X = np.hstack((X, omega[:, np.newaxis]))
    return X, y

def II_11_17(n=10):
    a, b = [1, 3]
    n_0 = (b-a)*np.random.random_sample(n) + a
    Ef = (b-a)*np.random.random_sample(n) + a
    p_d = (b-a)*np.random.random_sample(n) + a
    theta = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    y = n_0*(1+p_d*Ef*np.cos(theta)/(kb*T))
    X = np.hstack((n_0[:, np.newaxis], Ef[:, np.newaxis]))
    X = np.hstack((X, p_d[:, np.newaxis]))
    X = np.hstack((X, theta[:, np.newaxis]))
    X = np.hstack((X, kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    return X, y

def II_11_20(n=10):
    a, b = [1, 5]
    n_rho = (b-a)*np.random.random_sample(n) + a
    Ef = (b-a)*np.random.random_sample(n) + a
    p_d = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    y = n_rho*p_d**2*Ef/(3*kb*T)
    X = np.hstack((n_rho[:, np.newaxis], Ef[:, np.newaxis]))
    X = np.hstack((X, p_d[:, np.newaxis]))
    X = np.hstack((X, kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    return X, y

def II_11_27(num=100):
    a, b = [0, 1]
    n = (b-a)*np.random.random_sample(num) + a
    Ef = (b-a)*np.random.random_sample(num) + a
    alpha = (b-a)*np.random.random_sample(num) + a
    epsilon = (b-a)*np.random.random_sample(num) + a
    y = n*alpha/(1-(n*alpha/3))*epsilon*Ef
    X = np.hstack((n[:, np.newaxis], Ef[:, np.newaxis]))
    X = np.hstack((X, alpha[:, np.newaxis]))
    X = np.hstack((X, epsilon[:, np.newaxis]))
    return X, y

def II_11_28(num=100):
    a, b = [0, 1]
    n = (b-a)*np.random.random_sample(num) + a
    alpha = (b-a)*np.random.random_sample(num) + a
    X = np.hstack((n[:, np.newaxis], alpha[:, np.newaxis]))
    y = 1+n*alpha/(1-(n*alpha/3))
    return X, y

def II_13_17(n=10):
    a, b = [1, 5]
    c = (b-a)*np.random.random_sample(n) + a
    I = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    y = 1/(4*np.pi*epsilon*c**2)*2*I/r
    X =  np.hstack((c[:, np.newaxis], I[:, np.newaxis]))
    X = np.hstack((X, r[:, np.newaxis]))
    X = np.hstack((X, epsilon[:, np.newaxis]))
    return X, y

def II_13_23(n=100):
    a, b = [1, 5]
    rho_c_0 = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    v = (b-a)*np.random.random_sample(n) + a
    a, b = [3, 10]
    c = (b-a)*np.random.random_sample(n) + a
    y = rho_c_0/np.sqrt(1-v**2/c**2)
    X =  np.hstack((rho_c_0[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    return X, y

def II_13_34(n=10):
    a, b = [1, 5]
    rho_c_0 = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    v = (b-a)*np.random.random_sample(n) + a
    a, b = [3, 10]
    c = (b-a)*np.random.random_sample(n) + a
    X =  np.hstack((rho_c_0[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    y = rho_c_0*v/np.sqrt(1-v**2/c**2)
    return X, y

def II_5_4(n=10):
    a, b = [1, 5]
    mom = (b-a)*np.random.random_sample(n) + a
    B = (b-a)*np.random.random_sample(n) + a
    theta = (b-a)*np.random.random_sample(n) + a
    y = -mom*B*np.cos(theta)
    X =  np.hstack((mom[:, np.newaxis], B[:, np.newaxis]))
    X = np.hstack((X, theta[:, np.newaxis]))
    return X, y

def II_15_5(n=10):
    a, b = [1, 5]
    p_d = (b-a)*np.random.random_sample(n) + a
    Ef = (b-a)*np.random.random_sample(n) + a
    theta = (b-a)*np.random.random_sample(n) + a
    X =  np.hstack((p_d[:, np.newaxis], Ef[:, np.newaxis]))
    X = np.hstack((X, theta[:, np.newaxis]))
    y = -p_d*Ef*np.cos(theta)
    return X, y

def II_21_32(n=10):
    a, b = [1, 5]
    q = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    v = (b-a)*np.random.random_sample(n) + a
    a, b = [3, 10]
    c = (b-a)*np.random.random_sample(n) + a
    y =q/(4*np.pi*epsilon*r*(1-v/c))
    X = np.hstack((q[:, np.newaxis], r[:, np.newaxis]))
    X = np.hstack((X, epsilon[:, np.newaxis]))
    X = np.hstack((X, v[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    return X, y

def II_24_17(n=10):
    a, b = [4, 6]
    omega = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    c = (b-a)*np.random.random_sample(n) + a
    a, b = [2, 4]
    d = (b-a)*np.random.random_sample(n) + a
    y = np.sqrt(omega**2/c**2-np.pi**2/d**2)
    X = np.hstack((omega[:, np.newaxis], c[:, np.newaxis]))
    X = np.hstack((X, d[:, np.newaxis]))
    return X, y

def II_27_16(n=10):
    a, b = [1, 5]
    epsilon = (b-a)*np.random.random_sample(n) + a
    c = (b-a)*np.random.random_sample(n) + a
    Ef = (b-a)*np.random.random_sample(n) + a
    y = epsilon*c*Ef**2
    X = np.hstack((epsilon[:, np.newaxis], c[:, np.newaxis]))
    X = np.hstack((X, Ef[:, np.newaxis]))
    return X, y

def II_27_18(n=10):
    a, b = [1, 5]
    epsilon = (b-a)*np.random.random_sample(n) + a
    Ef = (b-a)*np.random.random_sample(n) + a
    y = epsilon*Ef**2
    X = np.hstack((epsilon[:, np.newaxis], Ef[:, np.newaxis]))
    return X, y

def II_34_2a(n=10):
    a, b = [1, 5]
    q = (b-a)*np.random.random_sample(n) + a
    v = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    y = q*v/(2*np.pi*r)
    X = np.hstack((q[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, r[:, np.newaxis]))
    return X, y

def II_34_2(n=10):
    a, b = [1, 5]
    q = (b-a)*np.random.random_sample(n) + a
    v = (b-a)*np.random.random_sample(n) + a
    r = (b-a)*np.random.random_sample(n) + a
    X = np.hstack((q[:, np.newaxis], v[:, np.newaxis]))
    X = np.hstack((X, r[:, np.newaxis]))
    y = q*v*r/2
    return X, y

def II_34_11(n=10):
    a, b = [1, 5]
    q = (b-a)*np.random.random_sample(n) + a
    g_ = (b-a)*np.random.random_sample(n) + a
    B = (b-a)*np.random.random_sample(n) + a
    m = (b-a)*np.random.random_sample(n) + a
    y = g_*q*B/(2*m)
    X = np.hstack((q[:, np.newaxis], g_[:, np.newaxis]))
    X = np.hstack((X, B[:, np.newaxis]))
    X = np.hstack((X, m[:, np.newaxis]))
    return X, y

def II_34_29a(n=10):
    a, b = [1, 5]
    q = (b-a)*np.random.random_sample(n) + a
    h = (b-a)*np.random.random_sample(n) + a
    m = (b-a)*np.random.random_sample(n) + a
    y = q*h/(4*np.pi*m)
    X = np.hstack((q[:, np.newaxis], h[:, np.newaxis]))
    X = np.hstack((X, m[:, np.newaxis]))
    return X, y

def II_34_29b(n=10):
    a, b = [1, 5]
    g_ = (b-a)*np.random.random_sample(n) + a
    mom = (b-a)*np.random.random_sample(n) + a
    B = (b-a)*np.random.random_sample(n) + a
    Jz = (b-a)*np.random.random_sample(n) + a
    h = (b-a)*np.random.random_sample(n) + a
    y = g_*mom*B*Jz/(h/(2*np.pi))
    X = np.hstack((g_[:, np.newaxis], mom[:, np.newaxis]))
    X = np.hstack((X, B[:, np.newaxis]))
    X = np.hstack((X, Jz[:, np.newaxis]))
    X = np.hstack((X, h[:, np.newaxis]))
    return X, y

def II_35_18(n=10):
    a, b = [1, 3]
    n_0 = (b-a)*np.random.random_sample(n) + a
    mom = (b-a)*np.random.random_sample(n) + a
    B = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    y = n_0/(np.exp(mom*B/(kb*T))+np.exp(-mom*B/(kb*T)))
    X = np.hstack((n_0[:, np.newaxis], mom[:, np.newaxis]))
    X = np.hstack((X, B[:, np.newaxis]))
    X = np.hstack((X, kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    return X, y

def II_35_21(n=10):
    a, b = [1, 5]
    n_rho = (b-a)*np.random.random_sample(n) + a
    mom = (b-a)*np.random.random_sample(n) + a
    B = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    y = n_rho*mom*np.tanh(mom*B/(kb*T))
    X = np.hstack((n_rho[:, np.newaxis], mom[:, np.newaxis]))
    X = np.hstack((X, B[:, np.newaxis]))
    X = np.hstack((X, kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    return X, y

def II_36_38(n=10):
    a, b = [1, 3]
    H = (b-a)*np.random.random_sample(n) + a
    mom = (b-a)*np.random.random_sample(n) + a
    alpha = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    epsilon = (b-a)*np.random.random_sample(n) + a
    c = (b-a)*np.random.random_sample(n) + a
    M = (b-a)*np.random.random_sample(n) + a
    y = mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M
    X = np.hstack((H[:, np.newaxis], mom[:, np.newaxis]))
    X = np.hstack((X, alpha[:, np.newaxis]))
    X = np.hstack((X, kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    X = np.hstack((X, epsilon[:, np.newaxis]))
    X = np.hstack((X, c[:, np.newaxis]))
    X = np.hstack((X, M[:, np.newaxis]))
    return X, y

def II_37_1(n=10):
    a, b = [1, 5]
    chi = (b-a)*np.random.random_sample(n) + a
    mom = (b-a)*np.random.random_sample(n) + a
    B = (b-a)*np.random.random_sample(n) + a
    y = mom*(1+chi)*B
    X = np.hstack((chi[:, np.newaxis], mom[:, np.newaxis]))
    X = np.hstack((X, B[:, np.newaxis]))
    return X, y

def II_38_3(n=10):
    a, b = [1, 5]
    Y = (b-a)*np.random.random_sample(n) + a
    A = (b-a)*np.random.random_sample(n) + a
    x = (b-a)*np.random.random_sample(n) + a
    d = (b-a)*np.random.random_sample(n) + a
    y = Y*A*x/d
    X = np.hstack((Y[:, np.newaxis], A[:, np.newaxis]))
    X = np.hstack((X, x[:, np.newaxis]))
    X = np.hstack((X, d[:, np.newaxis]))
    return X, y

def II_38_14(n=10):
    a, b = [1, 5]
    Y = (b-a)*np.random.random_sample(n) + a
    sigma = (b-a)*np.random.random_sample(n) + a
    y = Y/(2*(1+sigma))
    X = np.hstack((Y[:, np.newaxis], sigma[:, np.newaxis]))
    return X, y

def III_4_32(n=10):
    a, b = [1, 5]
    h = (b-a)*np.random.random_sample(n) + a
    omega = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    y = 1/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)
    X = np.hstack((h[:, np.newaxis], omega[:, np.newaxis]))
    X = np.hstack((X, kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    return X, y

def III_4_33(n=10):
    a, b = [1, 5]
    h = (b-a)*np.random.random_sample(n) + a
    omega = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    y = (h/(2*np.pi))*omega/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)
    X = np.hstack((h[:, np.newaxis], omega[:, np.newaxis]))
    X = np.hstack((X, kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    return X, y

def III_7_38(n=10):
    a, b = [1, 5]
    h = (b-a)*np.random.random_sample(n) + a
    mom = (b-a)*np.random.random_sample(n) + a
    B = (b-a)*np.random.random_sample(n) + a
    y = 2*mom*B/(h/(2*np.pi))
    X = np.hstack((h[:, np.newaxis], mom[:, np.newaxis]))
    X = np.hstack((X, B[:, np.newaxis]))
    return X, y
    
def III_8_54(n=10):
    a, b = [1, 2]
    E_n = (b-a)*np.random.random_sample(n) + a
    t = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 4]
    h = (b-a)*np.random.random_sample(n) + a
    y = np.sin(E_n*t/(h/(2*np.pi)))**2
    X = np.hstack((E_n[:, np.newaxis], t[:, np.newaxis]))
    X = np.hstack((X, h[:, np.newaxis]))
    return X, y

def III_9_52(n=10**3):
    a, b = [1, 3]
    p_d = (b-a)*np.random.random_sample(n) + a
    Ef = (b-a)*np.random.random_sample(n) + a
    t = (b-a)*np.random.random_sample(n) + a
    h = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 5]
    omega = (b-a)*np.random.random_sample(n) + a
    omega_0 = (b-a)*np.random.random_sample(n) + a
    y = (p_d*Ef*t/(h/(2*np.pi)))*np.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2
    X = np.hstack((p_d[:, np.newaxis], Ef[:, np.newaxis]))
    X = np.hstack((X, t[:, np.newaxis]))
    X = np.hstack((X, h[:, np.newaxis]))
    X = np.hstack((X, omega[:, np.newaxis]))
    X = np.hstack((X, omega_0[:, np.newaxis]))
    return X, y

def III_10_19(n=100):
    a, b = [1, 5]
    mom = (b-a)*np.random.random_sample(n) + a
    Bx = (b-a)*np.random.random_sample(n) + a
    By = (b-a)*np.random.random_sample(n) + a
    Bz = (b-a)*np.random.random_sample(n) + a
    y = mom*np.sqrt(Bx**2+By**2+Bz**2)
    X = np.hstack((mom[:, np.newaxis], Bx[:, np.newaxis]))
    X = np.hstack((X, By[:, np.newaxis]))
    X = np.hstack((X, Bz[:, np.newaxis]))
    return X, y

def III_12_43(num=10):
    a, b = [1, 5]
    n = (b-a)*np.random.random_sample(num) + a
    h = (b-a)*np.random.random_sample(num) + a
    y = n*(h/(2*np.pi))
    X = np.hstack((n[:, np.newaxis], h[:, np.newaxis]))
    return X, y

def III_13_18(n=10):
    a, b = [1, 5]
    E_n = (b-a)*np.random.random_sample(n) + a
    d = (b-a)*np.random.random_sample(n) + a
    k = (b-a)*np.random.random_sample(n) + a
    h = (b-a)*np.random.random_sample(n) + a
    y = 2*E_n*d**2*k/(h/(2*np.pi))
    X = np.hstack((E_n[:, np.newaxis], d[:, np.newaxis]))
    X = np.hstack((X, k[:, np.newaxis]))
    X = np.hstack((X, h[:, np.newaxis]))
    return X, y

def III_14_14(n=10):
    a, b = [1, 5]
    I_0 = (b-a)*np.random.random_sample(n) + a
    a, b = [1, 2]
    q = (b-a)*np.random.random_sample(n) + a
    Volt = (b-a)*np.random.random_sample(n) + a
    kb = (b-a)*np.random.random_sample(n) + a
    T = (b-a)*np.random.random_sample(n) + a
    y = I_0*(np.exp(q*Volt/(kb*T))-1)
    X = np.hstack((I_0[:, np.newaxis], q[:, np.newaxis]))
    X = np.hstack((X, Volt[:, np.newaxis]))
    X = np.hstack((X, kb[:, np.newaxis]))
    X = np.hstack((X, T[:, np.newaxis]))
    return X, y

def III_15_12(n=10):
    a, b = [1, 5]
    U = (b-a)*np.random.random_sample(n) + a
    k = (b-a)*np.random.random_sample(n) + a
    d = (b-a)*np.random.random_sample(n) + a
    y = 2*U*(1-np.cos(k*d))
    X = np.hstack((U[:, np.newaxis], k[:, np.newaxis]))
    X = np.hstack((X, d[:, np.newaxis]))
    return X, y

def III_15_14(n=10):
    a, b = [1, 5]
    h = (b-a)*np.random.random_sample(n) + a
    E_n = (b-a)*np.random.random_sample(n) + a
    d = (b-a)*np.random.random_sample(n) + a
    y = (h/(2*np.pi))**2/(2*E_n*d**2)
    X = np.hstack((h[:, np.newaxis], E_n[:, np.newaxis]))
    X = np.hstack((X, d[:, np.newaxis]))
    return X, y
    
def III_15_27(num=10):
    a, b = [1, 5]
    alpha = (b-a)*np.random.random_sample(num) + a
    n = (b-a)*np.random.random_sample(num) + a
    d = (b-a)*np.random.random_sample(num) + a
    y = 2*np.pi*alpha/(n*d)
    X = np.hstack((alpha[:, np.newaxis], n[:, np.newaxis]))
    X = np.hstack((X, d[:, np.newaxis]))
    return X, y

def III_17_37(n=10):
    a, b = [1, 5]
    beta = (b-a)*np.random.random_sample(n) + a
    alpha = (b-a)*np.random.random_sample(n) + a
    theta = (b-a)*np.random.random_sample(n) + a
    y = beta*(1+alpha*np.cos(theta))
    X = np.hstack((alpha[:, np.newaxis], beta[:, np.newaxis]))
    X = np.hstack((X, theta[:, np.newaxis]))
    return X, y

def III_19_51(num=10):
    a, b = [1, 5]
    m = (b-a)*np.random.random_sample(num) + a
    q = (b-a)*np.random.random_sample(num) + a
    epsilon = (b-a)*np.random.random_sample(num) + a
    h = (b-a)*np.random.random_sample(num) + a
    n = (b-a)*np.random.random_sample(num) + a
    y = -m*q**4/(2*(4*np.pi*epsilon)**2*(h/(2*np.pi))**2)*(1/n**2)
    X = np.hstack((m[:, np.newaxis], q[:, np.newaxis]))
    X = np.hstack((X, epsilon[:, np.newaxis]))
    X = np.hstack((X, h[:, np.newaxis]))
    X = np.hstack((X, n[:, np.newaxis]))
    return X, y

def III_21_20(n=10):
    a, b = [1, 5]
    rho_c_0 = (b-a)*np.random.random_sample(n) + a
    q = (b-a)*np.random.random_sample(n) + a
    A_vec = (b-a)*np.random.random_sample(n) + a
    m = (b-a)*np.random.random_sample(n) + a
    y = -rho_c_0*q*A_vec/m
    X = np.hstack((rho_c_0[:, np.newaxis], q[:, np.newaxis]))
    X = np.hstack((X, A_vec[:, np.newaxis]))
    X = np.hstack((X, m[:, np.newaxis]))
    return X, y


func_dict = {'I_6_2a': I_6_2a, 'I_6_20': I_6_20, 'I_6_2b': I_6_2b, 'I_8_14': I_8_14,
    'I_9_18': I_9_18, 'I_10_7': I_10_7, 'I_11_19': I_11_19, 'I_12_1': I_12_1,
    'I_12_2': I_12_1, 'I_12_4': I_12_4, 'I_12_5': I_12_5, 'I_12_11': I_12_11,
    'I_13_4': I_13_4, 'I_13_12': I_13_12, 'I_14_3': I_14_3, 'I_14_4': I_14_4,
    'I_15_3x': I_15_3x, 'I_15_3t': I_15_3t, 'I_15_1': I_15_10, 'I_16_6': I_16_6,
    'I_18_4': I_18_4, 'I_18_12': I_18_12, 'I_18_14': I_18_14, 'I_24_6': I_24_6,
    'I_25_13': I_25_13, 'I_26_2': I_26_2, 'I_27_6': I_27_6, 'I_29_4': I_29_4,
    'I_29_16': I_29_16, 'I_30_3': I_30_3, 'I_30_5': I_30_5, 'I_32_5': I_32_5,
    'I_32_17': I_32_17, 'I_34_8': I_34_8, 'I_34_10': I_34_10, 'I_34_14': I_34_14,
    'I_34_27': I_34_27, 'I_37_4': I_37_4, 'I_38_12': I_38_12, 'I_39_10': I_39_10,
    'I_39_11': I_39_11, 'I_39_22': I_39_22, 'I_40_10': I_40_10, 'I_41_16': I_41_16,
    'I_43_16': I_43_16, 'I_43_31': I_43_31, 'I_43_43': I_43_43, 'I_44_4': I_44_4,
    'I_47_23': I_47_23, 'I_48_2': I_48_20, 'I_50_26': I_50_26, 'II_2_42': II_2_42,
    'II_3_24': II_3_24, 'II_4_23': II_4_23, 'II_6_11': II_6_11, 'II_6_15a': II_6_15a,
    'II_6_15b': II_6_15b, 'II_8_7': II_8_7, 'II_8_31': II_8_31, 'II_10_9': II_10_9,
    'II_10_3': II_10_3, 'II_11_17': II_11_17, 'II_11_20': II_11_20, 'II_11_27': II_11_27,
    'II_11_28': II_11_28, 'II_13_17': II_13_17, 'II_13_34': II_13_34, 'II_5_4': II_5_4,
    'II_15_5': II_15_5, 'II_21_32': II_21_32, 'II_24_17': II_24_17, 'II_27_16': II_27_16,
    'II_27_18': II_27_18, 'II_34_2a': II_34_2a, 'II_34_2': II_34_2, 'II_34_11': II_34_11,
    'II_34_29a': II_34_29a, 'II_34_29b': II_34_29b, 'II_35_18': II_35_18, 'II_35_21': II_35_21,
    'II_36_38': II_36_38, 'II_37_1': II_37_1, 'II_38_3': II_38_3, 'II_38_14': II_38_14,
    'III_4_32': III_4_32, 'III_4_33': III_4_33, 'III_7_38': III_7_38, 'III_8_54': III_8_54,
    'III_9_52': III_9_52, 'III_10_19': III_10_19, 'III_12_43': III_12_43, 'III_13_18': III_13_18,
    'III_14_14': III_14_14, 'III_15_12': III_15_12, 'III_15_14': III_15_14, 'III_15_27': III_15_27,
    'III_17_37': III_17_37, 'III_21_20': III_21_20, 'III_19_51': III_19_51}

# functions up to 3 variables
s_func_dict = {'I_6_2a': I_6_2a, 'I_6_20': I_6_20, 'I_6_2b': I_6_2b, 'I_10_7': I_10_7, 'I_12_1': I_12_1,
               'I_12_4': I_12_4, 'I_12_5': I_12_5, 'I_14_3': I_14_3, 'I_15_10': I_15_10, 
               'I_16_6':I_16_6, 'I_18_12':I_18_12, 'I_25_13':I_25_13, 'I_26_2':I_26_2, 'I_27_6':I_27_6,
                'I_29_4':I_29_4, 'I_30_3':I_30_3,
                'I_30_5':I_30_5, 
                'I_34_10':I_34_10,
                'I_34_14': I_34_14,
                'I_34_27': I_34_27,
                'I_37_4': I_37_4,
                'I_39_10': I_39_10,
                'I_39_11': I_39_11,
                'I_43_31': I_43_31,
                'I_47_23': I_47_23,
                'I_48_20': I_48_20,
                'II_3_24': II_3_24,
                'II_4_23': II_4_23,
                'II_8_7': II_8_7,
                'II_8_31': II_8_31,
                'II_10_9':II_10_9,
                'II_11_28': II_11_28,
                'II_13_23': II_13_23,
                'II_13_34': II_13_34,
                'II_5_4': II_5_4,
                'II_15_5': II_15_5,
                'II_24_17': II_24_17,
                'II_27_16': II_27_16,
                'II_27_18': II_27_18,
                'II_34_2a': II_34_2a,
                'II_34_2': II_34_2,
                'II_34_29a': II_34_29a,
                'II_37_1': II_37_1,
                'II_38_14': II_38_14,
                'III_7_38': III_7_38,
                'III_8_54': III_8_54,
                'III_12_43': III_12_43,
                'III_15_12': III_15_12,
                'III_15_14': III_15_14,
                'III_15_27': III_15_27,
                'III_17_37': III_17_37
}
