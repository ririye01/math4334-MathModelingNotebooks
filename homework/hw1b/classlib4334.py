import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sint
from scipy import integrate
from scipy import optimize
from scipy import stats
from scipy import special



def fit_and_plot(fitfunc, x, y, p0=None, bounds=(-np.inf, np.inf), xlabel="x", ylabel="y", xlim=None, ylim=None, title="Model vs. Data", residual=False, filename=None):

    # get the fit and the error
    fit, cov = optimize.curve_fit(fitfunc, x, y, p0=p0, bounds=bounds)
    err = 2*np.sqrt(np.diag(cov))
    ypr = fitfunc(x, *fit)

    if residual:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 3.5))
    else:
        fig, ax1 = plt.subplots(1,1,figsize=(5, 3.5))
    
    # plot data and model
    ax1.plot(x, y, 's', label='data')
    ax1.plot(x, ypr, '-', label='model')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    if xlim is not None:  ax1.set_xlim(xlim)
    if ylim is not None:  ax1.set_ylim(ylim)
    ax1.legend(loc='best')

    # plot residual
    if residual:
        res = y - ypr
        
        ax2.plot(x, res, 's')
        ax2.plot(x, 0*x, 'k--')
        ax2.set_xlabel('x')
        ax2.set_ylabel('r')
        ax2.set_title('Residual')
        y1min, y1max = ax1.get_ylim()
        y1span = y1max-y1min
        ax2.set_ylim(-y1span/2, y1span/2)
   
    
    fig.tight_layout()
    if filename is not None:  plt.savefig(filename)
    

    # create the text report summarizing the fit
    fittext1 = ""
    
    # get the parameter names and report their values
    pnames = inspect.getfullargspec(fitfunc)[0][1:]
    #fittext += "Parameter Values: 95% confidence\n"
    #fittext += "\n"
    for (a, e, p) in zip(fit, err, pnames):
        fittext1 += "%4s = %8.5f +- %8.5f  (rel: %3.3f%%)\n" % (p, a, e, 100*e/np.abs(a))

    # calculate absolute / adjusted R-squared
    fittext2 = ""
    N = len(x)
    P = len(fit)
    yav = np.mean(y)
    ssres = np.sum((y-ypr)**2)
    sstot = np.sum((y-yav)**2)
    rsq   = 1 - ssres/sstot
    arsq  = 1 - ssres/sstot * (N-1)/(N-P)
    fittext2 += 'Unexplained Variance: %1.5f\n' % (1-rsq) 
    fittext2 += 'Explained Variance:   %1.5f' % (rsq) 

    # resize the figure for inline display
    if residual:
        fig.set_size_inches(10, 4.5)
        plt.subplots_adjust(bottom=0.3)
        plt.figtext(0.07, 0.18, fittext1, va='top', fontsize=10, fontfamily='monospace')
        plt.figtext(0.6, 0.18, fittext2, va='top', fontsize=10, fontfamily='monospace')
    else:
        fig.set_size_inches(5, 4.5)
        plt.subplots_adjust(bottom=0.3)
        plt.figtext(0.1, 0.18, fittext1, va='top', fontsize=10, fontfamily='monospace')
    plt.show()

    # return the fitted values and the uncertainties
    return fit, err



def advanced_residual_analysis(model, datax, data, params, vsypred=False, normalize=False):

    NPTS = len(data)
    NPRM = len(params)

    preds = model(datax, *params)

    resid = data - preds
    rmean = np.mean(resid)
    rstd  = np.std(resid)
    
    dmean = np.mean(data)
    ssres = np.sum((data - preds)**2)
    sstot = np.sum((data - dmean)**2)
    uevar = (ssres / (NPTS - NPRM)) / (sstot / (NPTS - 1))

    
    if normalize:
        nresids = (resid - rmean) / rstd
        rlabel  = 'normalized residual'
        rmean = 0
        rstd = 1
    else:
        nresids = resid
        rlabel  = 'residual'
    
    
    plt.figure(figsize=(9, 5.5))
    
    plt.subplot(221)
    if vsypred:
        plt.plot(preds, data, 's')
        plt.plot(preds, preds, 'k--', lw=1.5)
        plt.xlabel('predicted y')
        plt.ylabel('actual y')
        plt.title('Predicted vs Actual')
    else:
        plt.plot(datax, data, 's')
        plt.plot(datax, preds, '-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Model vs Data')
        ymin,ymax = plt.gca().get_ylim()
        

    
    plt.subplot(222)
    if vsypred:
        plt.plot(preds, nresids, 's')
        plt.plot(preds, 0*preds, 'k--')
        plt.xlabel('predicted y')
    else:
        plt.plot(datax, nresids, 's')
        plt.plot(datax, 0*datax, 'k--')
        plt.xlabel('x')

    if normalize:
        plt.ylim(-4*rstd, 4*rstd)
    else:
        plt.ylim(-2*np.std(data), 2*np.std(data))
        

    plt.ylabel(rlabel)

    plt.title('Adjusted UVR = %6.4f' % (uevar))
    
    bincount = int(np.floor(2*np.sqrt(len(resid))))
    binedges = np.linspace(rmean-4*rstd, rmean+4*rstd, bincount+1)
    binwidth = rstd*7/bincount
    
    plt.subplot(223)
    plt.hist(nresids, binedges, width=0.85*binwidth)
    plt.xlim(binedges[0], binedges[-1])
    plt.xlabel(rlabel)
    plt.ylabel('frequency')
    plt.title('Histogram of Residuals')
    
    plt.subplot(224)
    
    rsort = sorted(nresids)
    percents = (np.arange(NPTS) + 0.5) / NPTS
    resids_p = rmean + rstd * np.sqrt(2) * special.erfinv(2*percents - 1)


    plt.plot(rsort, resids_p, 's')
    plt.plot(rsort, rsort, 'k--')
    plt.xlim(binedges[0], binedges[-1])
    plt.xlabel(rlabel)
    plt.ylabel('predicted')
    plt.title('Expected/Actual Residuals')
    
    plt.tight_layout()
    plt.show()

    return 









def compare_nested(x, y, simplefunc, complexfunc, simpleguess=None, complexguess=None, bounds=(-np.inf, np.inf), plot=False):

    N = len(x)

    # perform the simple fit and get SSRES
    sfit, scov = optimize.curve_fit(simplefunc, x, y, p0=simpleguess, bounds=bounds)
    serr = np.sqrt(np.diag(scov))
    sypr = simplefunc(x, *sfit)
    sres = sypr - y
    sdof = N - len(sfit)
    sssres = np.sum(sres**2)

    # perform the complex fit and get SSRES
    cfit, ccov = optimize.curve_fit(complexfunc, x, y, p0=complexguess, bounds=bounds)
    cerr = np.sqrt(np.diag(ccov))
    cypr = complexfunc(x, *cfit)
    cres = cypr - y
    cdof = N - len(cfit)
    cssres = np.sum(cres**2)

    # calculate F-number and get P-value
    fnum = ((sssres - cssres) / cssres)   /   ((sdof - cdof) / cdof)
    pval = stats.f.sf(fnum, sdof-cdof, cdof)
    
    
    # plot?
    if plot == True:
        plt.figure(figsize=(10,3.5))

        plt.subplot(121)
        plt.plot(x, y, 'ks')
        plt.plot(x, sypr, '-', label='simple')
        plt.plot(x, cypr, '-', label='complex')
        plt.legend(loc='best')
        #plt.xlabel(xlabel)
        #plt.ylabel(ylabel)
        #plt.title(datatitle)

        plt.subplot(122)
        plt.plot(x, 0*x, 'k--')
        plt.plot(x, sres, 's', label='simple')
        plt.plot(x, cres, 's', label='complex')
        plt.legend(loc='best')
        #plt.ylim(-1.2*sramax, 1.2*sramax)
        #plt.xlabel(xlabel)
        #plt.ylabel('r')
        #plt.title(restitle)

        plt.tight_layout()
        plt.show()
    
    
    
    
    # report
    mstring = "complex" if pval < .05 else "simple"
    print('F-test comparing %s (simple) vs. %s (complex)' % (simplefunc.__name__, complexfunc.__name__))
    print('')
    print('  Simple:   SS=%8f, DOF=%4d' % (sssres, sdof) )
    print('  Complex:  SS=%8f, DOF=%4d' % (cssres, cdof) )
    print('  F-number: %1.4f ' % (fnum) )
    print('  P-value:  %1.4f ' % (pval) )
    print('  Random?   %1.3f%%' % (pval*100))
    print('')
    print('It is recommended to prefer the %s function.' % (mstring))
    print('')

    return fnum, pval






def fit_and_plot_old(fitfunc, x, y, p0=None, bounds=(-np.inf, np.inf), plots=True, scaleres=None, xlabel="x", ylabel="y", xlim=None, ylim=None, datatitle="Model vs. Data", restitle="Relative Residual", filename=None):

    # get the fit and the error
    fit, cov = optimize.curve_fit(fitfunc, x, y, p0=p0, bounds=bounds)
    err = 2*np.sqrt(np.diag(cov))
    ypr = fitfunc(x, *fit)
    res = y - ypr

    # plot data and model, together with residual
    if (plots):
        plt.figure(figsize=(10,3.5))

        plt.subplot(121)
        plt.plot(x, y, 'bs')
        plt.plot(x, ypr, 'r')
        if xlim!=None:  plt.xlim(xlim)
        if ylim!=None:  plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(datatitle)

        residual_scaling = 1.0
        restitle = 'Residual'
        resylabel = 'residual [raw]'
        if scaleres == 'std':
            residual_scaling = np.stdev(res)
            restitle = 'Residual (scaled by std)'
            resylabel = 'residual [std. devs.]'
        if scaleres == 'rms':
            residual_scaling = np.sqrt(np.mean(y**2)) * .01
            restitle = 'Residual (scaled by rms)'
            resylabel = 'residual [%]'
        if scaleres == 'model':
            residual_scaling = ypr
            restitle = 'Residual (scaled by model)'
            resylabel = 'residual [%]'
        scaled_residual = res / residual_scaling * .01
        sramax = np.max(np.abs(scaled_residual))
        
        plt.subplot(122)
        plt.plot(x, 0*x, 'k--')
        plt.plot(x, scaled_residual, 'bs')
        plt.ylim(-1.2*sramax, 1.2*sramax)
        plt.xlabel(xlabel)
        plt.ylabel(resylabel)
        plt.title(restitle)

        plt.tight_layout()
        if filename != None:
            plt.savefig(filename)
        plt.show()

    # get the parameter names and report their values
    pnames = inspect.getfullargspec(fitfunc)[0][1:]
    print("Parameter Values: 95%")
    print("")
    for (a, e, p) in zip(fit, err, pnames):
        print("%4s = %12.12f +- %12.12f  (rel: %3.3f%%)" % (p, a, e, 100*e/np.abs(a)))
    print("")

    # calculate and display absolute / adjusted R-squared
    N = len(x)
    P = len(fit)
    yav = np.mean(y)
    ssres = np.sum((y-ypr)**2)
    sstot = np.sum((y-yav)**2)
    rsq   = 1 - ssres/sstot
    arsq  = 1 - ssres/sstot * (N-1)/(N-P)
    print('absolute r-squared: %1.8f  (%4.2f nines)' % (rsq, -np.log10(1-rsq)) )
    print('adjusted r-squared: %1.8f  (%4.2f nines)' % (arsq, -np.log10(1-arsq)) )
    print('\n')

    # return the fitted values and the uncertainties
    return fit, err




def numerical_phase_line(tvals, xvals, labels=None, sg_window=5, sg_degree=2, filename=None):
    
    # smooth data and derivative using Savitsky-Golay
    from scipy.signal import savgol_filter
    
    N = len(tvals)

    d0t = []
    d0x = []
    d1x = []
    
    for kk in range(N):
        dt = tvals[kk][1] - tvals[kk][0]
        d0t.append( savgol_filter(tvals[kk], sg_window, sg_degree, deriv=0, delta=dt) )
        d0x.append( savgol_filter(xvals[kk], sg_window, sg_degree, deriv=0, delta=dt) )
        d1x.append( savgol_filter(xvals[kk], sg_window, sg_degree, deriv=1, delta=dt) )

        
    # sort the pairs d0x, d1x by the value of d0x
    pairs = []
    for d0s, d1s in zip(d0x, d1x):
        pairs.extend( [[d0, d1] for d0, d1 in zip(d0s, d1s)] )
        
    spairs = sorted(pairs, key=lambda x: x[0])
    sd0x = np.array([p[0] for p in spairs])
    sd1x = np.array([p[1] for p in spairs])
        

        
    plt.figure(figsize=(6,3.5))

    for kk in range(N):
        if labels == None:
            plt.plot(d0x[kk], d1x[kk], 's')
        else:
            plt.plot(d0x[kk], d1x[kk], 's', label=labels[kk])
            plt.legend(loc='best')

    plt.plot([sd0x[0], sd0x[-1]], [0, 0], 'k--')
    plt.title('Smoothed Phase Line')
    plt.xlabel('x')
    plt.ylabel('dx/dt')
    plt.tight_layout()
    
    if filename is not None: plt.savefig(filename)
    
    plt.show()

    return sd0x, sd1x





def ode_fit_and_plot(oderhs, t, xdata, p0, bounds=None, plots=True, xlabel="t", ylabel="x", datatitle="Model vs. Data", restitle="Relative Residual"):

    xdata = np.array(xdata)
    xN = len(xdata)
    
    pnames = inspect.getfullargspec(oderhs)[0][2:]
    pN = len(pnames)
    
    IC = [a[0] for a in xdata]

    def model(t, params):
        result = integrate.solve_ivp(oderhs, (t[0], t[-1]), IC, t_eval=t, args=params)
        if result.success:
            return result['y']
        else:
            print(params)
            mstring = "Within classlib4334.ode_fit_and_plot(), scipy.integrate.solve_ode() failed."
            mstring += "\n\n"
            mstring += "Message from solve_ode():  " + result.message
            mstring += "\n\n"
            mstring += "This can occur if the optimizer happens to try parameter values for which the ODE 'blows up.'  Can you find a better guess?"
            raise Exception(mstring)

    def resid(params):
        xmodel = model(t, params)
        return xdata - xmodel

    def ssres(params):
        myres = resid(params)
        return np.linalg.norm(myres)

    pguess = p0
    print(pguess)
    result = optimize.minimize(ssres, pguess, method='L-BFGS-B', bounds=bounds, options=None)  # not returning an inverse hessian?
    fit    = result.x
    ihess  = result.hess_inv.todense()
    resvar = np.var(resid(fit), ddof=len(pguess))
    cov    = ihess*resvar*2.0
    #print(ihess)
    #print(cov)
    
    
    # get the fit and the error
    err = 2*np.sqrt(np.diag(cov))
    xpr = model(t, fit)
    res = resid(fit)

    # used in displaying residual
    rms = np.sqrt(np.mean(xdata**2))
    rmx = np.max(np.abs(res/rms))


    # plot data and model, together with residual
    if (plots):

        plt.figure(figsize=(10,3.5))
     
        # obtain standard color cycle values
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        # plot data and fits in standard color cycle
        plt.subplot(121)
        for ii in range(xN):
            plt.plot(t, xdata[ii], color=colors[ii], marker='s', linestyle='', label='data %d' % (ii+1))
            plt.plot(t, xpr[ii], color=colors[ii], marker='', linestyle='-')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.title(datatitle)

        # plot residuals together with standard color cycles
        plt.subplot(122)
        plt.plot(t, 0*t, 'k--')
        for ii in range(xN):
            plt.plot(t, res[ii]/rms, color=colors[ii], marker='s', linestyle='', label='data %d' % (ii+1))      # get this to occur in standard color cycle
        plt.ylim(-1.2*rmx, 1.2*rmx)
        plt.xlabel(xlabel)
        plt.ylabel('r')
        plt.legend(loc='best')
        plt.title(restitle)

        plt.tight_layout()
        plt.show()

        
    # get the parameter names and report their values
    pnames = inspect.getfullargspec(oderhs)[0][2:]
    print("Parameter Values: 95%")
    print("")
    for (a, e, p) in zip(fit, err, pnames):
        print("%4s = %10.6f +- %1.6f" % (p, a, e))
    print("")

    # calculate and display absolute / adjusted R-squared
    N = len(t)
    P = len(fit)
    xav = np.mean(xdata)
    ssres = np.sum((xdata-xpr)**2)
    sstot = np.sum((xdata-xav)**2)
    rsq   = 1 - ssres/sstot
    arsq  = 1 - ssres/sstot * (N-1)/(N-P)
    print('absolute r-squared: %1.8f  (%4.2f nines)' % (rsq, -np.log10(1-rsq)) )
    print('adjusted r-squared: %1.8f  (%4.2f nines)' % (arsq, -np.log10(1-arsq)) )
    print('\n')

    # return the fitted values and the uncertainties
    return fit, err