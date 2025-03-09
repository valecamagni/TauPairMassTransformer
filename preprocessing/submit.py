from optparse import OptionParser 
import os 

infiles = [os.getcwd()+'/DY_H_TT_SUSY140-350_semilep.txt']

files = [line.strip() for t in infiles for line in open(t) if not line.strip().startswith('#') and line.strip()]

print("Input has" , len(files) , "files")

for idx, file in enumerate(files):

    parts = file.split('/')
    sample = parts[-2]  
    folder = sample

    # Create the directories (if needed)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.getcwd()+'/'+folder+'/JOBS', exist_ok=True)
    os.makedirs(os.getcwd()+'/'+folder+'/DATA', exist_ok=True)
    os.makedirs(os.getcwd()+'/'+folder+'/PLOTS', exist_ok=True)
    os.makedirs(os.getcwd()+'/'+folder+'/LOGS', exist_ok=True)

    outJobName =  os.getcwd()+'/'+folder+'/JOBS/job_' + str(idx) + '.sh'
    inListName =  os.getcwd()+'/'+folder+'/JOBS/filelist_' + str(idx) + '.txt'
    jobfilelist = open(inListName, 'w')
    jobfilelist.write(file+"\n")
    jobfilelist.close()

    flat = int(1)
    pairType2 = "mu_tau"

    #command = "python3 " + "fully_hadronic.py" + " -i " + file + " -f " + str(flat) 
    command1 = "python3 " + "semi_leptonic.py" + " -i " + file + " -f " + str(flat) # default: ele_tau
    command2 = "python3 " + "semi_leptonic.py" + " -i " + file + " -f " + str(flat) + " -p " + pairType2

    
    scriptFile = open (outJobName, 'w')
    scriptFile.write ('#!/bin/bash\n') 
    scriptFile.write ('echo $HOSTNAME\n')
    scriptFile.write ('cd %s\n' % os.getcwd())
    scriptFile.write ('export KRB5CCNAME=/gwpool/users/camagni/krb5cc_`id -u camagni`\n')      
    scriptFile.write ('export X509_USER_PROXY=~/x509up_u`id -u $USER`\n')      
    scriptFile.write ('eosfusebind\n')
    scriptFile.write ('source ~gennai/.bashrc\n')
    scriptFile.write ('mamba activate torch\n')
    scriptFile.write (command1+'\n')
    scriptFile.write (command2+'\n')
    scriptFile.close()

    os.system ('chmod u+rwx ' + outJobName) 

    condorFile = open ('%s/condorLauncher_%d.sh'% (os.getcwd()+'/'+folder+'/JOBS',idx), 'w')
    condorFile.write ('Universe = vanilla\n')
    condorFile.write ('Executable  = '+outJobName +'\n')
    condorFile.write ('Log         = condor_job_$(ProcId).log\n')
    condorFile.write ('Output      = condor_job_$(ProcId).out\n')
    condorFile.write ('Error       = condor_job_$(ProcId).error\n')
    condorFile.write ('getenv       = True\n')
    condorFile.write ('Requirements = (machine == "pccms02.hcms.it") || (machine == "pccms12.hcms.it") || (machine == "pccms13.hcms.it")\n')
    condorFile.write ('queue 1\n')
    condorFile.close ()

    command = 'condor_submit '+ os.getcwd()+'/'+folder+'/JOBS' + '/condorLauncher_' + str(idx) + '.sh'
    print(command)
    os.system (command) 