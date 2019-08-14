"""
1, demonstrate cases when it converges or not converge
2, plot the phase picture
"""

"""
X(N*d), x_i~normal(0,I_d)
R(w)=\|w\|_1(2)^2
g(X,w)=D(w)Xw, D(w)=diag(Xw>0)
E(w)=\frac{1}{2N}\|g(X,w^*)-g(X,w)\|_2^2+\frac{\lambda}{2}R(w)
delta(w)=\frac{1}{N}X^T D(w)(D*Xw*-D(w)Xw)+\frac{\lambda}{2}R'(w)

Experiments:
1, examine prob and theoretical prob
  change \lambda in 0.001, 0.01, 0.1
  change N in 10, 20, 100
  change d in 2, 3, 5
2, plot converge and not converge

When it comes to l1 loss, judge convergence with another rule
"""
import numpy as np
import matplotlib.pyplot as plt

def show(N=20, d=2, lamb=0.01, eta=0.05, epsilon=0.1):
    X = np.zeros([N, d])
    for i in range(N):
        for j in range(d):
            X[i, j] = np.random.normal()
    wstar = np.ones(d)
            
    def R(w):
        return np.linalg.norm(w) ** 2
        #return np.sum(np.abs(w))
    
    def dR(w):
        return 2 * w
        #return np.sign(w)
    
    
    def D(w):
        N = np.shape(X)[0]
        A = np.zeros([N, N])
        for i in range(N):
            A[i, i] = (np.dot(X, w)[i] > 0) + .0
        return A
        
    def g(w):
        temp = np.dot(D(w), X)
        return np.dot(temp, w)
        
    def E(w):
        # E(w)=\frac{1}{2N}\|g(X,w^*)-g(X,w)\|_2^2+\frac{\lambda}{2}R(w)
        return 1.0/(2*N)*np.linalg.norm(g(wstar)-g(w))**2 + lamb/2*R(w)
    
    def delta(w):
        # delta(w)=\frac{1}{N}X^T D(w)(D*Xw*-D(w)Xw)+\frac{\lambda}{2}R'(w)
        temp = np.dot(np.transpose(X), D(w))
        return 1.0/N*np.dot(temp, g(wstar)-g(w))+lamb/2*dR(w)
         
    def r(epsilon):
        return epsilon * np.linalg.norm(wstar) * np.sqrt(2*np.pi/(1+d))
    
    def bi(w):
        return np.dot(np.transpose(X), D(w)).dot(X)
    
    def w_hat(w):
        # w_hat = (X^TD(wstar)X-\lambda N Id)^{-1} X^TD(wstar)X wstar
        temp1 = np.matrix(bi(w) - lamb * N * np.eye(d)).I
        temp2 = np.dot(bi(w), w)
        return np.dot(temp1, temp2)
    
    def A(n, k):
        # probability of rank(D*)\leq k
        res = 1  # n choose 0
        for i in range(1, k+1):
            res += (np.math.factorial(n)+.0)/(np.math.factorial(i)*np.math.factorial(n-i))
        return res / (2 ** n)
    
    print 'prob =', (1 - epsilon) / 2 * (1 - A(N, d))
    wh = w_hat(wstar)
    print 'w_hat =', np.array(wh)[0]
    plt.plot(np.array(wh)[0][0], np.array(wh)[0][1], color='black', marker='^')
    #plt.axis([-0.6, 1.5, -1, 1.5])
    """rand in ball"""
    #w0 = np.random.rand(d) * r(epsilon)
    w0 = np.zeros(d)
    for i in range(d):
        l2 = np.linalg.norm(w0) ** 2
        rand = 2 * np.random.rand() - 1
        w0[i] = rand * np.sqrt(1 - l2)
    w0 = w0 * r(epsilon)
    # false initiation
    w0 = np.array([-0.5, -0.8])
    w = w0
    for num in range(3000):
        plt.plot(w[0], w[1], color='black', marker='o', markersize=2)
        
        if np.linalg.norm(w_hat(wstar)-wstar)>d/2:
            is_converge = False
            break
            
        #if num>1000 and np.linalg.norm(w_hat(wstar)-w)>d/2:
        if num>1000 and np.linalg.norm(wstar-w)>d:
            is_converge = False
            break
            
        w_old = w
        w += eta * delta(w)
        if num%1000 == 0:
            print w
        if np.linalg.norm(delta(w))<0.05*d/2 and np.linalg.norm(w-wstar)<0.01*d/2:
            print num, w
            is_converge = True
            break
        is_converge = False
    
    plt.show()
    return is_converge


def phase(N=20, d=2, lamb=0.01, eta=0.05, epsilon=0.1):
    X = np.zeros([N, d])
    for i in range(N):
        for j in range(d):
            X[i, j] = np.random.normal()
    wstar = np.ones(d)
    plt.plot(wstar[0], wstar[1], color='black', marker='o')
            
    def R(w):
        #return np.linalg.norm(w) ** 2
        return np.sum(np.abs(w))
    
    def dR(w):
        #return 2 * w
        return np.sign(w)
    
    
    def D(w):
        N = np.shape(X)[0]
        A = np.zeros([N, N])
        for i in range(N):
            A[i, i] = (np.dot(X, w)[i] > 0) + .0
        return A
        
    def g(w):
        temp = np.dot(D(w), X)
        return np.dot(temp, w)
        
    def E(w):
        # E(w)=\frac{1}{2N}\|g(X,w^*)-g(X,w)\|_2^2+\frac{\lambda}{2}R(w)
        return 1.0/(2*N)*np.linalg.norm(g(wstar)-g(w))**2 + lamb/2*R(w)
    
    def delta(w):
        # delta(w)=\frac{1}{N}X^T D(w)(D*Xw*-D(w)Xw)+\frac{\lambda}{2}R'(w)
        temp = np.dot(np.transpose(X), D(w))
        return 1.0/N*np.dot(temp, g(wstar)-g(w))+lamb/2*dR(w)
         
    def r(epsilon):
        return epsilon * np.linalg.norm(wstar) * np.sqrt(2*np.pi/(1+d))
    
    def bi(w):
        return np.dot(np.transpose(X), D(w)).dot(X)
    
    def w_hat(w):
        # w_hat = (X^TD(wstar)X-\lambda N Id)^{-1} X^TD(wstar)X wstar
        temp1 = np.matrix(bi(w) - lamb * N * np.eye(d)).I
        temp2 = np.dot(bi(w), w)
        return np.dot(temp1, temp2)
    
    def A(n, k):
        # probability of rank(D*)\leq k
        res = 1  # n choose 0
        for i in range(1, k+1):
            res += (np.math.factorial(n)+.0)/(np.math.factorial(i)*np.math.factorial(n-i))
        return res / (2 ** n)
    
    # phase
    for x in np.linspace(-1, 1.4, 13):
        for y in np.linspace(-1, 1.4, 13):
            plt.plot(x, y, color='black', marker='o', markersize=2)
            w = np.array([x, y])
            l = np.linalg.norm(delta(w))
            plt.plot([x, x+0.1/l*delta(w)[0]], [y, y+0.1/l*delta(w)[1]], color='green')
            #plt.plot(x+0.1/l*delta(w)[0], y+0.1/l*delta(w)[1], color='black', marker='^', markersize=3)
    plt.axis([-1.1, 1.5, -1.1, 1.5])
    plt.show()
    
phase(N=10)
    