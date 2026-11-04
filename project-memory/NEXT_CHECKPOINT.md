# Exact next checkpoint — approved EdNet core campaign

Current user approved replacement and fifteen missing five-seed core trainings.
CPU/data checks pass;prepared fixed manifest is
/workspace/LightSAKT2-architecture/experiments/ednet_replacement_20261104/plan.json.
First inspect reports/ednet_replacement_20261104/campaign_status.json and owned
processes. If absent and prepared commit/admission pass, launch ONCE:
CUDA_VISIBLE_DEVICES=MIG-d2fcfd79-720e-5218-82c9-0a132be68920 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u -m scripts.run_ednet_campaign_20261104
Redirect output to that report folder/training.log. Do not restart ifexisting.
Corebudget15serialwindows<=900s/4GiB each,total<=13500s,noautomaticretry. Stop if
unsafe;preserve latest/best and report reason. Latest checkpoints includeoptimizer,
all RNG and loaderorder state. Any continuation needs exact config/source identity
and remaining applicable budget;never reset or silently replace a run.

Data/protocol:EdNetKT1_5k_first,5000randomstudentfiles(seed20261104),firstpresented
question per solving session before answer filtering;authenticquestions/keys,tags
composites,studentsplit42,common d64/h2/dropout.1/lr.001/b64/L100/50max/patience5.
SixCPUtests andrealselflabel/heldoutleakage audit PASS. No test in training.
All15configs frozen. TestonlyCPU after each validation-selected checkpoint and
onlywith unchanged configs/splits. No furthermodelchanges/newsearch basedontest.

Aftercompleted runs:CPU replaybest validation ANDtest;verifycheckpointSHA/config,
allactual metrics,exactmask/targets,finiteoptimizer/RNG/dataorder andfeature/data
hashes. Preservepredictions andcheckpointbackups. Combine new results withunchanged
ASSIST2009/Algebra2005 five-seed records;omitASSIST2015 from finalthree,retainhistory.
MakeCSV means/sampleSD,individuals,pairedabsoluteAUCdeltas,graphs,professorreport,
preprocessing/source/split/reprodocs andverifiedportablearchive. Reportnegatives.
Use EdNet subset name explicitly;offline/richer-input/first-question limitations.

Prepared optionalSAKTQuestion controls are five configs underprepared_controls/;
CPU preparegenericadapter,do notdelaycoreorchargecore15 budgetforcontrols.
HiTSKT isONHOLD:do notread,modify,resume,run orsignalanystuff there. NopeerGPU
interference. Keepoldscopeholdbytes,oldresults/main dirtywork/Githistory intact.
Updateandstrictlyvalidatecanonicalmemory after milestones andbeforeending.
