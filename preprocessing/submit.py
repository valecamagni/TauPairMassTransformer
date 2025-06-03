from optparse import OptionParser 
import os 

base_folder = '/gwdata/users/camagni/DiTau/TPMT_DATA/'

infiles = [os.getcwd()+'/TTToSemiLeptonic.txt']

files = [line.strip() for t in infiles for line in open(t) if not line.strip().startswith('#') and line.strip()]

print("Input has" , len(files) , "files")

for idx, file in enumerate(files):

    parts = file.split('/')
    sample = parts[-2]  
    
    # Create the directories (if needed)
    os.makedirs(base_folder+sample, exist_ok=True)
    os.makedirs(base_folder+sample+'/JOBS', exist_ok=True)
    os.makedirs(base_folder+sample+'/DATA', exist_ok=True)
    os.makedirs(base_folder+sample+'/PLOTS', exist_ok=True)
    os.makedirs(base_folder+sample+'/LOGS', exist_ok=True)

    outJobName =  base_folder+sample+'/JOBS/job_' + str(idx) + '.sh'
    inListName =  base_folder+sample+'/JOBS/filelist_' + str(idx) + '.txt'
    jobfilelist = open(inListName, 'w')
    jobfilelist.write(file+"\n")
    jobfilelist.close()

    #command1 = "python3 " + "fully_hadronic.py" + " -i " + file + " -t false"
    command2 = "python3 " + "semi_leptonic.py" + " -i " + file + " -t false" # default: ele_tau
    command3 = "python3 " + "semi_leptonic.py" + " -i " + file + " -p mu_tau" + " -t false"

    #command1 = "python3 " + "fully_hadronic.py" + " -i " + file 
    #command2 = "python3 " + "semi_leptonic.py" + " -i " + file
    #command3 = "python3 " + "semi_leptonic.py" + " -i " + file +  " -p mu_tau"
    
    scriptFile = open (outJobName, 'w')
    scriptFile.write ('#!/bin/bash\n') 
    scriptFile.write ('echo $HOSTNAME\n')
    scriptFile.write ('cd %s\n' % os.getcwd())
    scriptFile.write ('export KRB5CCNAME=/gwpool/users/camagni/krb5cc_`id -u camagni`\n')      
    scriptFile.write ('export X509_USER_PROXY=~/x509up_u`id -u $USER`\n')      
    scriptFile.write ('eosfusebind\n')
    scriptFile.write ('source ~gennai/.bashrc\n')
    scriptFile.write ('mamba activate torch\n')
    #scriptFile.write (command1+'\n')
    scriptFile.write (command2+'\n')
    scriptFile.write (command3+'\n')
    scriptFile.close()

    os.system ('chmod u+rwx ' + outJobName) 

    condorFile = open ('%s/condorLauncher_%d.sh'% (base_folder+sample+'/JOBS',idx), 'w')
    condorFile.write ('Universe = vanilla\n')
    condorFile.write ('Executable  = '+outJobName +'\n')
    condorFile.write ('Log         = condor_job_$(ProcId).log\n')
    condorFile.write ('Output      = condor_job_$(ProcId).out\n')
    condorFile.write ('Error       = condor_job_$(ProcId).error\n')
    condorFile.write ('getenv       = True\n')
    condorFile.write ('Requirements = (machine == "pccms01.hcms.it") || (machine == "pccms02.hcms.it")\n')
    condorFile.write ('queue 1\n')
    condorFile.close ()

    command = 'condor_submit '+ base_folder + sample +'/JOBS' + '/condorLauncher_' + str(idx) + '.sh'
    print(command)
    os.system (command) 