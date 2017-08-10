def powerfitfn(x,pars):
    ''' Power fit:
        A*x**B + C
    '''
    if(hasattr(pars['A'],"value")):
        A = pars['A'].value
        B = pars['B'].value
        C = pars['C'].value
    else:
        A = pars['A']
        B = pars['B']
        C = pars['C']
    return A*x**B + C
