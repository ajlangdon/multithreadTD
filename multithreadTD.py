# Single- and multi-thread TD functions
#
# Code for model simulations in
# Takahashi, Y.K., Stalnaker, T.A., Mueller, L.E. et al.
# Dopaminergic prediction errors in the ventral tegmental
# area reflect a multithreaded predictive model.
# Nat Neurosci (2023).
# https://doi.org/10.1038/s41593-023-01310-x
#
# A. Langdon July 2021


import numpy as np
import numpy.matlib
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import seaborn as sns

def TDlambda(events, T, O, taskstates, reward, eta, gamma, elambda, decay, init_w=0):

    """
    TD learning (single thread)

    :events: vector ntrials x ntimepoints containing codes for task events during each trial
    :T: matrix nstates x nstates P(state_{t+1}|state_{t})
    :O: matrix nstates x nstates x nobs for P(o_{t+1}|state_{t},state_{t+1}) OR nstates x nobs for P(o_{t}|state_{t})
    :taskstates: maxtrix nstates x ntaskstates of mapping for states into (macro) taskstates [0,1]
    :reward: nobs of reward amount per observation
    :eta: learning rate 0<=eta<=1
    :gamma: discount 0<=discount<=1
    :elambda: eligibility 0<=elambda<=1
    :decay: trial-by-trial decay in weights 0<=decay<=1
    """

    nstates = T.shape[0] # number of states
    ntr = events.shape[0] # number of trials
    t_max = events.shape[1] # number of timepoints
    ts_max = taskstates.shape[1] # number of taskstates

    # states
    init_state = np.zeros(nstates)
    init_state[0] = 1 # start in background on each trial

    # save states and values
    state = np.zeros([ntr,nstates,t_max])
    nextstate = np.zeros([ntr,nstates,t_max])
    weights = np.zeros([ntr,nstates,t_max])
    eligibility = np.zeros([ntr,nstates,t_max])
    value = np.zeros([ntr,t_max])
    nextvalue = np.zeros([ntr,t_max])
    rpe = np.zeros([ntr,t_max])
    activethread = np.zeros([ntr,ts_max,t_max])

    w = np.zeros(nstates) + init_w # evolving weights

    for tr in range(ntr):

        # initial state is always background pre-cue
        s = init_state
        bgprev = init_state
        e = np.zeros(nstates) # evolving eligibility traces *within trial*
        thread = np.zeros(ts_max) # evolving taskstate occupancy
        all_thread = np.zeros(ts_max) # all threads that have been active in the trial

        for t in range(t_max-1):

            # extract current value from state representation
            v = w@s # current value
            # calculate next state
            if O.ndim<3:
                sprime = O[:,events[tr,t+1]]*(s@T)
            else:
                sprime = s@(T*O[:,:,events[tr,t+1]])
            sprime = sprime/sprime.sum() # normalize
            # extract next state value
            vprime = w@sprime

            # first decay eligibility traces of previous states
            e = gamma*elambda*e
            # update eligibility with current state
            e = np.clip(e + s,0,1) # these can't be >1 for any state that we re-enter (e.g. background)
            # for consistency, let's zero out the eligibility of the bg
            e[0] = 0

            # eligibility is erased for any taskstate that is not occupied
            # tracking current thread at the level of taskstates
            thread = s@taskstates
            # eligibility is limited by thread occupancy
            for ts in range(ts_max):
                e[taskstates[:,ts]] = e[taskstates[:,ts]]*thread[ts]

            # update all threads
            all_thread = np.maximum(all_thread,thread)

            # learning for state at t
            delta = reward[events[tr,t+1]] + gamma*vprime - v
            # weight update
            w = w + eta*e*delta

            # but we assume the background does not accrue predictions
            w[0] = 0

            state[tr,:,t] = s # save belief states
            nextstate[tr,:,t] = sprime
            weights[tr,:,t] = w # and weights
            eligibility[tr,:,t] = e # and eligibility traces
            value[tr,t] = v # timepoint values
            nextvalue[tr,t] = vprime
            activethread[tr,:,t] = thread

            rpe[tr,t] = delta # and rpe

            bgprev = s[0]
            s = sprime # transition to next state

        # trial-to-trial loss in threads that were active
        w = (1-decay*(taskstates@all_thread))*w

    return state, nextstate, weights, eligibility, value, nextvalue, rpe, activethread


def TDlambda_vectorErrorTupdate(events, T, O, M, U, taskstates, reward, eta, gamma, elambda, decay, t_eta, stream_start, init_w=0):
    """
    vector TD learning (multithread) with transition updates

    :events: vector ntrials x ntimepoints containing codes for task events during each trial
    :T: matrix nstates x nstates P(state_{t+1}|state_{t})
    :O: matrix nstates x nstates x nobs for P(o_{t+1}|state_{t},state_{t+1}) OR nstates x nobs for P(o_{t}|state_{t})
    :M: maxtrix nids x nstates of mapping from states to vector prediction dimension
    :U: matrix nstates x nids of mapping from vector rpe dimension to states (nb we use M.T)
    :taskstates: maxtrix nstates x ntaskstates of mapping for states into (macro) taskstates [0,1]
    :reward: nids x nobs of reward amount per observation for each dim
    :eta: learning rate 0<=eta<=1
    :gamma: discount 0<=discount<=1
    :elambda: eligibility 0<=elambda<=1
    :decay: trial-by-trial decay in weights 0<=decay<=1
    :init_w: initial weights
    """
    # values are vectorized, RPE is also vectorized
    # M controls the dimensionality of the prediction errors
    # i.e. maps from timepoint states to discrete 'channels' of prediction

    # we also have updates of the transition matrix according to observations in a trial

    # here, reward should be nids x nobs

    nstates = T.shape[0] # number of states
    ntr = events.shape[0] # number of trials
    t_max = events.shape[1] # number of timepoints
    nids = M.shape[0] # dimensionality of the prediction
    # check orientation of taskstates
    if taskstates.shape[1]>taskstates.shape[0]:
        taskstates = taskstates.T
    ntstates = taskstates.shape[1] # number of taskstates
    # print(type(taskstates[0,0]))

    # states
    init_state = np.zeros(nstates)
    init_state[0] = 1 # start in background on each trial

    # save states and values
    state = np.zeros([ntr,nstates,t_max])
    nextstate = np.zeros([ntr,nstates,nstates,t_max]) # this is the full progression
    weights = np.zeros([ntr,nstates,t_max])
    eligibility = np.zeros([ntr,nstates,t_max])
    value = np.zeros([ntr,nids,t_max])
    nextvalue = np.zeros([ntr,nids,t_max])
    rpe = np.zeros([ntr,nids,t_max])
    id = np.zeros([ntr,nids,t_max])
    activethread = np.zeros([ntr,ntstates,t_max])

    w = np.zeros(nstates) + init_w # evolving weights

    # and evolving transition matrix
    tmatrix = np.zeros([ntr,nstates,nstates])
    tmatrix[0,:,:] = T

    # indices of the initial transition into the taskstates
    # stream_start = np.array([1,81,161],dtype='int')

    for tr in range(ntr):

        # initial state is always background pre-cue
        s = init_state
        e = np.zeros(nstates) # evolving eligibility traces *within trial*
        thread = np.zeros(ntstates) # evolving taskstate occupancy

        # outcome tracking for transition learning
        tr_outcome = np.zeros([nids])

        # thread activity
        threadhistory = np.zeros([ntstates])
        # print(f'threadhistory {threadhistory.shape}')

        for t in range(t_max-1):
            # print(t)

            # track next obs
            if events[tr,t+1]>1:
                tr_outcome[events[tr,t+1]-2] = 1

            # extract current value from state representation
            v = M@(w*s) # current value of current state
            # calculate values of next states *within channels*
            smap = np.reshape(s,[nstates,1]) # reshape operates in place so lets reassign
            sproj = np.matlib.repmat(smap,1,nstates) # projection of current state to right dimensions (i.e. explicit transpose)
            if len(np.shape(O))==2:
                oproj = np.matlib.repmat(np.reshape(O[:,events[tr,t+1]],[1,nstates]),nstates,1) # the obs P that goes with sprime
            elif len(np.shape(O))==3:
                oproj = O[:,:,events[tr,t+1]] # the obs P that goes with sprime conditional on s
            else:
                print('dim error in O')

            sprime = sproj*(tmatrix[tr,:,:]*oproj)
            sprime = sprime/np.sum(sprime[:]) # normalize
            # compute next state value *within channel*
            vprime = M@sprime@w

            # tracking current thread at the level of taskstates
            # print(s.shape),print(taskstates.shape),print(threadhistory.shape)
            thread = s@taskstates
            # tracking thread history
            threadhistory = np.maximum(threadhistory,thread)

            # first decay eligibility traces of previous states
            e = gamma*elambda*e
            # update eligibility with current state
            e = np.clip(e + s,0,1) # these can't be >1 for any state that we re-enter (e.g. background)
            # for consistency, let's zero out the eligibility of the bg
            e[0] = 0
            # eligibility is limited by thread occupancy
            # note eligibility update was already proportional to belief
            for ts in range(ntstates):
                e[taskstates[:,ts]] = e[taskstates[:,ts]]*(thread[ts]>0)


            # learning for state at t
            # prediction error should be vector of length nids
            # in the appropriate ID channel
            delta = reward[:,events[tr,t+1]] + gamma*vprime - v
            # state weights are updated by these prediction errors according to channel membership
            w = w + eta*e*(U@delta) # update weights in state space
            # but we assume the background does not accrue predictions
            w[0] = 0

            state[tr,:,t] = s # save belief states
            nextstate[tr,:,:,t] = sprime
            weights[tr,:,t] = w # and weights
            eligibility[tr,:,t] = e # and eligibility traces
            value[tr,:,t] = v # timepoint values
            nextvalue[tr,:,t] = vprime
            activethread[tr,:,t] = thread

            rpe[tr,:,t] = delta # and rpe

            s = sprime.sum(axis=0) # transition to next state


        # trial-to-trial loss in threads that were active
        w = (1-decay*(taskstates@threadhistory))*w

        # let's update the transition matrix at the end of the trial
        # normalize the observed transitions to make it P
        tr_outcome = tr_outcome/tr_outcome.sum()

        if tr < (ntr-1):
            tmatrix[tr+1,:,:] = T # keep all transitions
            # learn transitions only for the starting states of each thread
            for n in range(nids):
                if tr_outcome[n]>0: # if this transition happened
                    tmatrix[tr+1,0,stream_start[n]] = tmatrix[tr,0,stream_start[n]] + t_eta*(tr_outcome[n] - tmatrix[tr,0,stream_start[n]])
                else: # this transition didn't happen
                    tmatrix[tr+1,0,stream_start[n]] = tmatrix[tr,0,stream_start[n]]*(1-t_eta)

    return state, nextstate, weights, eligibility, value, nextvalue, rpe, activethread, tmatrix


def createTask(dt,t_max,ntr,blocktr):
    """
    Generate task events
    """

    t_cue = int(1/dt)
    t_terminal = t_cue + int(5/dt)

    # cue == 1, terminal reward == 2, chocolate == 3, vanilla == 4

    well1 = np.zeros([ntr,t_max],dtype='int')
    well1[:,t_cue] = 1
    well1[:,t_terminal] = 2
    well2 = np.copy(well1)

    #------ well1
    # block 1 - early chocolate
    well1[:blocktr,t_cue+int(0.5/dt)] = 3
    # block 2 - late vanilla
    well1[blocktr,t_cue+int(1/dt)] = 4
    well1[blocktr+1,t_cue+int(2/dt)] = 4
    well1[(blocktr+2):blocktr*2,t_cue+int(3/dt)] = 4
    # block 3 - early vanilla
    well1[(blocktr*2):blocktr*3,t_cue+int(0.5/dt)] = 4
    # block 4 - late vanilla
    well1[(blocktr*3),t_cue+int(1/dt)] = 4
    well1[(blocktr*3)+1,t_cue+int(2/dt)] = 4
    well1[((blocktr*3)+2):blocktr*4,t_cue+int(3/dt)] = 4
    # block 5 - early chocolate
    well1[blocktr*4:,t_cue+int(0.5/dt)] = 3


    #------ well2
    # block 1 - late vanilla
    well2[:blocktr,t_cue+int(3/dt)] = 4
    # block 2 - early chocolate
    well2[blocktr:blocktr*2,t_cue+int(0.5/dt)] = 3
    # block 3 - late chocolate
    well2[blocktr*2,t_cue+int(1/dt)] = 3
    well2[blocktr*2+1,t_cue+int(2/dt)] = 3
    well2[(blocktr*2+2):blocktr*3,t_cue+int(3/dt)] = 3
    # block 4 - early chocolate
    well2[blocktr*3:blocktr*4,t_cue+int(0.5/dt)] = 3
    # block 5 - late vanilla
    well2[blocktr*4,t_cue+int(1/dt)] = 4
    well2[blocktr*4+1,t_cue+int(2/dt)] = 4
    well2[(blocktr*4+2):,t_cue+int(3/dt)] = 4

    return well1, well2


def buildWorldModel(model,chains,bisect=12):

    # observations are the same for all models
    nobs = 5
    # 0 = null, 1 = cue, 2 = terminal, 3 = chocolate, 4 = vanilla

    # will assume all states are contiguous
    states = np.concatenate(chains)
    nstates = len(states)
    taskstates = np.arange(len(chains))
    ntaskstates = len(taskstates)

    bg = chains[0][0] # chain[0] is the bg and is only 1 state

    if model==4: # sequential reset TD with short and long terminal states
        # transition matrix
        T = np.zeros([nstates,nstates])
        # bg -> bg OR first state in chain 1
        T[bg,[chains[0][0],chains[1][0]]] = 0.5
        # chain 1 progression within chain
        T[chains[1][0:-1],chains[1][1:]] = 0.5
        # chain 1 first half transition to second chain
        T[chains[1][0:bisect],chains[2][0]] = 0.5
        # chain 1 second half transition to chain 3
        T[chains[1][bisect:-1],chains[3][0]] = 0.5
        # final state in chain 1 transition to bg
        T[chains[1][-1],bg] = 1
        # chain 2 progression within chain
        T[chains[2][0:-1],chains[2][1:]] = 0.5
        # or to background
        T[chains[2][0:-1],bg] = 0.5
        # final state in chain to bg
        T[chains[2][-1],bg] = 1
        # chain 3 progression within chain
        T[chains[3][0:-1],chains[3][1:]] = 0.5
        # or to background
        T[chains[3][0:-1],bg] = 0.5
        # final state in chain to bg
        T[chains[3][-1],bg] = 1

        # observation matrix
        O = np.zeros([nstates,nstates,nobs])

        ou_self = 0.5 # P(null| self->self transition)
        # this has to be the same for all self-> self transitions under the null observation
        # *with* same P(self->self) to avoid dynamic readjustment of the state occupancy from
        # the normalization

        # null/water/choc/vanilla == bg->bg
        O[bg,bg,0] = ou_self
        O[bg,bg,2] = (1-ou_self)/3
        O[bg,bg,3] = (1-ou_self)/3
        O[bg,bg,4] = (1-ou_self)/3

        # cue occurs on bg -> chain 1 transition
        O[bg,chains[1][0],1] = 1

        # null == within chain progression
        O[chains[1][0]:chains[1][-1],chains[1][1]:(chains[1][-1]+1),0] = np.identity(len(chains[1])-1)*ou_self
        # null == within chain progression
        O[chains[2][0]:chains[2][-1],chains[2][1]:(chains[2][-1]+1),0] = np.identity(len(chains[2])-1)*ou_self
        # null == within chain progression
        O[chains[3][0]:chains[3][-1],chains[3][1]:(chains[3][-1]+1),0] = np.identity(len(chains[3])-1)*ou_self
        # any reward before bisection initiates the second stream
        O[chains[1][:bisect],chains[2][0],2] = 1/3
        O[chains[1][:bisect],chains[2][0],3] = 1/3
        O[chains[1][:bisect],chains[2][0],4] = 1/3
        # any reward after bisection initiates the third stream
        O[chains[1][bisect:-1],chains[3][0],2] = 1/3
        O[chains[1][bisect:-1],chains[3][0],3] = 1/3
        O[chains[1][bisect:-1],chains[3][0],4] = 1/3
        # last state of chain 1 transitions to bg under null
        O[chains[1][-1],bg,0] = 1
        # reward in second chain == transition to bg
        O[chains[2][0:-1],bg,2] = 1/3
        O[chains[2][0:-1],bg,3] = 1/3
        O[chains[2][0:-1],bg,4] = 1/3
        # last state of chain 2 transitions to bg under null or reward
        O[chains[2][-1],bg,0] = 1/4
        O[chains[2][-1],bg,2] = 1/4
        O[chains[2][-1],bg,3] = 1/4
        O[chains[2][-1],bg,4] = 1/4
        # reward in third chain == transition to bg
        O[chains[3][0:-1],bg,2] = 1/3
        O[chains[3][0:-1],bg,3] = 1/3
        O[chains[3][0:-1],bg,4] = 1/3
        # last state of chain 3 transitions to bg under null or reward
        O[chains[3][-1],bg,0] = 1/4
        O[chains[3][-1],bg,2] = 1/4
        O[chains[3][-1],bg,3] = 1/4
        O[chains[3][-1],bg,4] = 1/4

        # taskstates group the chains
        taskstates = np.zeros([nstates,ntaskstates],dtype='bool') # must be boolean for the indexing I used
        for c in range(ntaskstates):
            taskstates[chains[c],c] = True

    elif model==5: # three parallel chains
        # transition matrix
        T = np.zeros([nstates,nstates])
        # bg -> bg OR first state in chain 1,2,3
        T[0,0] = 0.5 # background to self
        T[0,[chains[1][0],chains[2][0],chains[3][0]]] = (1-0.5)/3
        # chain 1 progression within chain
        T[chains[1][0:-1],chains[1][1:]] = 0.5
        # chain 1 transition to bg
        T[chains[1][0:-1],bg] = 0.5
        # final state in chain 1 transition to bg
        T[chains[1][-1],bg] = 1
        # chain 2 progression within chain
        T[chains[2][0:-1],chains[2][1:]] = 0.5
        # or to background
        T[chains[2][0:-1],bg] = 0.5
        # final state in chain to bg
        T[chains[2][-1],bg] = 1
        # chain 3 progression within chain
        T[chains[3][0:-1],chains[3][1:]] = 0.5
        # or to background
        T[chains[3][0:-1],bg] = 0.5
        # final state in chain to bg
        T[chains[3][-1],bg] = 1


        # 3D observation matrix for transition specific observation probabilities
        O = np.zeros([nstates,nstates,nobs])

        ou_self = 0.5 # P(null| self->self transition)
        # this has to be the same for all self-> self transitions under the null observation
        # *with* same P(self->self) to avoid dynamic readjustment of the state occupancy from
        # the normalization

        # null/water/choc/vanilla == bg->bg
        O[bg,bg,0] = ou_self
        O[bg,bg,2] = (1-ou_self)/3
        O[bg,bg,3] = (1-ou_self)/3
        O[bg,bg,4] = (1-ou_self)/3

        # cue == bg->start of each stream
        O[bg,[chains[1][0],chains[2][0],chains[3][0]],1] = 1/3

        # null == within chain 1,2,3 progression
        O[chains[1][0]:chains[1][-1],chains[1][1]:(chains[1][-1]+1),0] = np.identity(len(chains[1])-1)*ou_self
        O[chains[2][0]:chains[2][-1],chains[2][1]:(chains[2][-1]+1),0] = np.identity(len(chains[2])-1)*ou_self
        O[chains[3][0]:chains[3][-1],chains[3][1]:(chains[3][-1]+1),0] = np.identity(len(chains[3])-1)*ou_self
        # last state of each chain transitions to bg under null
        O[chains[1][-1],bg,0] = 1
        O[chains[2][-1],bg,0] = 1
        O[chains[3][-1],bg,0] = 1

        # now the reward observations control the reset behavior within stream
        # single stream reset for each of these rewards
        # water == stream 1 -> bg
        O[chains[1][0]:chains[1][-1],bg,2] = 1
        # choc == stream 2 -> bg
        O[chains[2][0]:chains[2][-1],bg,3] = 1
        # vanilla == stream 3 -> bg
        O[chains[3][0]:chains[3][-1],bg,4] = 1

        # other reward observations allow progression
        # water allows continuation of streams 2 and 3
        O[chains[2][0]:chains[2][-1],chains[2][1]:(chains[2][-1]+1),2] = np.identity(len(chains[2])-1)
        O[chains[3][0]:chains[3][-1],chains[3][1]:(chains[3][-1]+1),2] = np.identity(len(chains[3])-1)
        # choc allows continuation of stream 1 and 3
        O[chains[1][0]:chains[1][-1],chains[1][1]:(chains[1][-1]+1),3] = np.identity(len(chains[1])-1)
        O[chains[3][0]:chains[3][-1],chains[3][1]:(chains[3][-1]+1),3] = np.identity(len(chains[3])-1)
        # vanilla allows continuation of stream 1 and 2
        O[chains[1][0]:chains[1][-1],chains[1][1]:(chains[1][-1]+1),4] = np.identity(len(chains[1])-1)
        O[chains[2][0]:chains[2][-1],chains[2][1]:(chains[2][-1]+1),4] = np.identity(len(chains[2])-1)


        # taskstates group the chains
        taskstates = np.zeros([nstates,ntaskstates],dtype='bool') # must be boolean for the indexing I use
        for c in range(ntaskstates):
            taskstates[chains[c],c] = True

    else:
        print(f"model {model} not found...")
        T = []
        O = []
        return

    return T,O,taskstates

def createTransitionMatrix(nstates,model,nid):
    """
    Generate transition matrix for a given simple TD model
    nstates: number of states in a thread
    model:  1 - TD(lambda) without reset
            2 - TD(lambda) with global reset
            3 - TD(lamdba) with sequential reset
    nid: number of separate threads/unique outcomes (for sequential reset models)
    """

    if model==1:
        # build transition for model 1 == single chain, no reset
        T = np.zeros([nstates,nstates])
        T[0,0] = 0.5 # background to self
        T[0,1] = 0.5 # background to chain
        T[1:-1,2:] = np.identity(nstates-2) # each chain state transitions to the next
        T[-1,0] = 1 # final chain state transitions to background

        return T

    elif model==2:
        # model 2 == single chain, with reset
        # includes a possible transition to the background upon observing a reward
        T = np.zeros([nstates,nstates])
        T[0,0] = 0.5 # background to self
        T[0,1] = 0.5 # background to chain
        T[1:-1,2:] = np.identity((nstates-2))*0.5 # each chain state transitions to the next or to the background
        T[1:-1,0] = 0.5
        T[-1,0] = 1 # final chain state transitions to background completely

        return T


    elif model==3:
        # sequential single stream predictions
        T = np.zeros([(nstates-1)*nid + 1, (nstates-1)*nid + 1])
        T[0,0] = 0.5 # background to self
        T[0,1] = 1-0.5 # background to first prediction

        # temporal progression within first prediction
        T[1:(nstates-1),2:nstates] = np.identity(nstates-2)*0.5
        # or first prediction transitions to second
        T[1:(nstates-1),nstates] = 0.5
        # final state of first prediction goes to second
        T[(nstates-1),nstates] = 1

        # temporal progression within second prediction
        T[nstates:((nstates-1)*2),((nstates-1)+2):((nstates-1)*2+1)] = np.identity(nstates-2)*0.5
        # or to background
        T[nstates:((nstates-1)*2),0] = 0.5
        # final state of second prediction goes to background
        T[(nstates-1)*2,0] = 1

        return T

    else:
        print(f"model {model} not found...")
        return

def createObservationMatrix3D(nstates,model,nid,bisect=4):
    # observations are:
    # 0 == null/nothing
    # 1 == cue
    # 2 == terminal reward
    # 3 == chocolate
    # 4 == vanilla
    # (see also task matrix)

    nobs = 5

    if model==1:
        # 3D observation matrix with no reset, single thread
        O = np.zeros([nstates,nstates,nobs])

        O[0,0,0] = 1/4 # null == bg->bg
        O[1:-1,2:,0] = np.identity(nstates-2)/4 # null == within state progression
        O[0,1,1] = 1 # cue == bg-> state 1
        O[0,0,1] = 0
        O[0,0,2] = 1/4
        O[0,0,3] = 1/4
        O[0,0,4] = 1/4
        O[1:-1,2:,2] = np.identity(nstates-2)/4
        O[1:-1,2:,3] = np.identity(nstates-2)/4
        O[1:-1,2:,4] = np.identity(nstates-2)/4
        O[-1,0,0] = 1

        return O

    elif model==2:
        # 3D observation matrix with reset, single stream
        O = np.zeros([nstates,nstates,nobs])

        O[0,0,0] = 1/4 # null == bg->bg
        O[1:-1,2:,0] = np.identity(nstates-2) # null == within state progression
        O[0,1,1] = 1 # cue == bg-> state 1
        # O[0,0,1] = 0 # no cue in bg
        O[0,0,2] = 1/4
        O[0,0,3] = 1/4
        O[0,0,4] = 1/4
        O[1:,0,2] = 1/3
        O[1:,0,3] = 1/3
        O[1:,0,4] = 1/3

        return O

    elif model==3:
        # 3D observation matrix with reset to next sequential single prediction stream
        O = np.zeros([(nstates-1)*nid + 1,(nstates-1)*nid + 1,nobs])

        ou_self = 0.5 # P(null| self->self transition)

        # null/water/choc/vanilla == bg->bg
        O[0,0,0] = ou_self
        O[0,0,2] = (1-ou_self)/3
        O[0,0,3] = (1-ou_self)/3
        O[0,0,4] = (1-ou_self)/3

        # cue == bg->start of first prediction
        O[0,1,1] = 1

        # first prediction
        # null == within stream progression
        O[1:(nstates-1),2:nstates,0] = np.identity(nstates-2)*ou_self
        # next sequential prediction - can only see null or water
        O[nstates:((nstates-1)*2),(nstates+1):nstates*2-1,0] = np.identity(nstates-2)*ou_self
        # and the terminal state for first prediction transition to second under null observation
        O[nstates-1,nstates,0] = 1# and the terminal state for first prediction transition to second under null observation
        # and terminal state of second prediction transitions to bg under null
        O[(nstates-1)*2,0,0] = 1

        # and any flavored reward observation in first prediction prompts
        # initiation of the second sequential prediction
        # water == stream 1 -> stream 2
        O[1:(nstates-1),nstates,2] = 1/3
        # choc == stream 1 -> stream 2
        O[1:(nstates-1),nstates,3] = 1/3
        # vanilla == stream 1 -> stream 2
        O[1:(nstates-1),nstates,4] = 1/3

        # any reward in second prediction == transition to bg
        O[nstates:(nstates*2-2),0,2] = 1/3
        O[nstates:(nstates*2-2),0,3] = 1/3
        O[nstates:(nstates*2-2),0,4] = 1/3

        return O

    else:
        print(f"model {model} not found...")
        return



# EOF
