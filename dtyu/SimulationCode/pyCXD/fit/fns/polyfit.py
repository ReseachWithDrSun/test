def polyfitfn(x,pars):
    ''' Polynomial fit'''
    alst = []
    if(hasattr(pars['n'],"value")):
        n = pars['n'].value
        for i in range(n+1):
            alst.append(pars['a{:d}'.format(i)].value)
    else:
        n = pars['n']
        for i in range(n+1):
            alst.append(pars['a{:d}'.format(i)])

    result = np.zeros(len(x))
    for i in range(len(alst)):
        res += alst[i]*x**i

