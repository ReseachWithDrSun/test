from multiprocessing import Process, Queue, Lock, Value

#-----------Parallel Programming Routines------------------
def runjobs(jobstuff,setupvars):
    ''' runjobs(jobstuff,setupvars) 
        Run by nodes. Look for jobs in the jobqueue. When there
        are none left, just return and stop.
        job stuff is a tuple of the following:
            ID - the id given to the job by master
            jq - the job queue
            rq - the result queue
            dq - the done queue, just a friendly way to tell master process that process will kill itself
            l  - the lock (to access the queue, i had issues before)
        setupvars is the tuple given by the program for setup of the data.
            It is just passed to jobcode.
    '''
    ID,jq,rq,dq,l = jobstuff

    # Prog dependent code
    # set up the initial data necessary for the code (to save time)
    # done setup
    #setup code: calc vars that should disappear when code is done
    # main loop, look for jobs, and keep running
    while(True):
        # grab job from queue (no lock necessary this is process safe), then run
        # add result to result queue
        # if no job, then just return
        # queue code
        try:
            l.acquire()
            if(jq.empty() is False):
                jqvars = jq.get(False,2)
                l.release()
                # run the code
                result = process_job(jqvars,setupvars)
                # put the result to the queue
                rq.put(result)
            else:
                l.release()
                time.sleep(2)
                print("Queue is empty so waiting 2 secs")
        except queue.Empty:
            print("Warning. Queue was empty. Trying again...")

        
def runmaster(numprocs, jobsetup, setupvars, results):
    ''' Run the master process that will run everything else.
        numprocs - number of extra processes to spawn
        jobsetup - tuple to describe the job setup
        varsetup - tuple to describe the variable setup
        requires these functions:
            init_results(varsetup) - returns a results object or tuple etc
            add_job(jobsetup) - iterator, returns a jobvar
            run_job(jobvar,varsetup) - runs a job described by jobvar
                returns a result
            process_result(resultvar, result) - processes a result with 
                and saves into result
            finalize_results(result,varsetup) - finalizes results
                for example, write result to a file etc

        internal functions:
            add_result(jobvar,resultvar) - adds the result (internal)
            
    '''

    #create the job queue
    jq = Queue()
    jobtot = 0
    for jvar in add_job(jobsetup):
        jq.put(jvar)
        jobtot += 1

    time.sleep(1)#sleep a second to allow job queue to be ready

    # create the run queue with the results
    rq = Queue()
    # create the done queue (to double check everything went well)
    dq = Queue()

    # create a lock
    l = Lock()

    # keep pointer to processes (not used though)
    procs = []
    #create the processes
    for i in range(numprocs):
        jobstuff = i,jq,rq,dq,l
        p = Process(target=runjobs,args=(jobstuff,setupvars))
        procs.append(p)

    for proc in procs:
        proc.start()

    # keep count of total results and stop when all acquired
    cnt = 0
    # master process, keep checking queue
    while(True):
        result = rq.get()

        process_result(result,results)
        cnt += 1
        time.sleep(1)

        print("Got job {} of {}".format(cnt,jobtot))
        if(cnt == jobtot):
            print("Master finished, exiting and terminating processes...")
            break

    print("Master finished, exiting and terminating processes...")
    #kill all processes
    for proc in procs:
        proc.terminate()

#-----------End of parallel programming routines-------------------

