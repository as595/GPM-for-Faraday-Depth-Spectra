import numpy as np


p1 = [100.,100.,100.,25.,25.,25.,25.,25.,25.,25.,25.,25.,25.,0.0,1.80,0.0,1.80]
phi1=[500.10,49.38,4.96,-37.84,-37.84,-37.84,-44.55,232.56,-37.83,-37.84,149.50,-232.56,-44.55,0.0,-240.22,0.0,-240.0]
chi1=[40.,60.,60.,0.,-40.,-40.,0.0,40.,-40.,0.0,40.,0.0,0.0,0.0,-36.,0.,-36.]

p2 = [0.0,0.0,0.0,16.70,24.0,9.0,16.70,9.0,16.50,9.0,23.75,9.0,24.0,0.0,0.0,0.0,0.0]
phi2=[0.0,0.0,0.0,103.18,5.05,5.05,37.50,192.70,5.05,103.0,163.50,-50.10,37.54,0.0,0.0,0.0,0.0]
chi2=[0.0,0.0,0.0,-36.,-40.,-40.,72.,40.,140.,-36.,-68.,72.,72.,0.0,0.0,0.0,0.0]

p3 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.9,1.9,1.9,1.9]
phic=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-136.98,-250.17,-136.98,-250.17]
phis=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,50.,50.,25.,25.]

def faraday_thin(p0,phi,chi,l2):

    chi*=np.pi/180.
    p = p0*np.exp(2*1j*(chi+phi*l2))

    return p

def faraday_thick(p0,phi_c,phi_s,l2):

    p = p0*(np.sin(phi_s*l2)/(phi_s*l2))*np.exp(2*1j*(phi_c*l2+0.5*phi_s*l2))
    
    return p

def make_model(model,l2):

    p = 0+1j*0
    if p1[model]>0.:
        p += faraday_thin(p1[model],phi1[model],chi1[model],l2)
    
    if p2[model]>0.:
        p += faraday_thin(p2[model],phi2[model],chi2[model],l2)

    if p3[model]>0.:
        p += faraday_thick(p3[model],phic[model],phis[model],l2)

    return p

def calc_chi2_r(p1,p2,var):
    
    chi2 = np.sum(np.abs(p1-p2)**2/var)
    chi2/= (2*len(p1)-4)
    
    return chi2


def calc_rm_wtd(dspec,model):

    phi = phi1[model]
    rm_wtd = np.sum(np.abs(dspec)*phi)/np.sum(np.abs(dspec))

    return rm_wtd


def calc_smse(p1,p2,var):

    q_pred = np.real(p1)
    u_pred = np.imag(p1)
    
    q_true = np.real(p2)
    u_true = np.imag(p2)
    
    qdiff = (q_pred-q_true)**2/var
    udiff = (u_pred-u_true)**2/var
    
    smse = (np.sum(qdiff)+np.sum(udiff))/(2*len(qdiff))

    return smse


def calc_sll(p1,p2,gp_var):
    
    """
    p1 - true
    p2 - gp prediction
    Equation 2.34 http://www.gaussianprocess.org/gpml/chapters/RW.pdf
    """
    
    # SLL:
    nll1_q = (np.real(p1)-np.real(p2))**2/(2*gp_var)
    nll1_q += 0.5*np.log(2.*np.pi*gp_var)
    nll1_u = (np.imag(p1)-np.imag(p2))**2/(2*gp_var)
    nll1_u += 0.5*np.log(2.*np.pi*gp_var)
    
    # normalisation
    q_mean = np.mean(np.real(p1))
    q_std  = np.std(np.real(p1))
    u_mean = np.mean(np.imag(p1))
    u_std  = np.std(np.imag(p1))
                    
    q_samp = np.random.normal(q_mean,q_std,size=len(p1))
    u_samp = np.random.normal(u_mean,u_std,size=len(p1))
    
    nll2_q = (np.real(p1)-q_samp)**2/(2*q_std)
    nll2_q += 0.5*np.log(2.*np.pi*q_std)
    nll2_u = (np.imag(p1)-u_samp)**2/(2*u_std)
    nll2_u += 0.5*np.log(2.*np.pi*u_std)
    
    # MSLL:
    nll_q = nll1_q - nll2_q
    nll_u = nll1_u - nll2_u
    
    sll = np.hstack((nll_q,nll_u))
    msll = np.mean(sll)
    
    return msll
